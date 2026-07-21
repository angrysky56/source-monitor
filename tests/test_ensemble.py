"""CPU tests for the F23 low-rank perturbation ensemble (no model download)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from source_monitor.llm.loop.ensemble import perturbed
from source_monitor.llm.loop.f22_ensemble import Arm, _evaluate, auroc


class _TinyNet(nn.Module):
    """Two linears named like a decoder layer so the suffix matcher applies."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.o_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.o_proj(x)))


def test_sigma_zero_is_a_true_noop() -> None:
    """The F23 control: sigma=0 must be bit-identical, not merely close."""
    net, x = _TinyNet(), torch.randn(4, 32)
    base = net(x)
    with perturbed(net, sigma=0.0) as n:
        assert n == 0
        assert torch.equal(net(x), base)


def test_perturbation_changes_output_then_restores_exactly() -> None:
    net, x = _TinyNet(), torch.randn(4, 32)
    base = net(x).clone()
    with perturbed(net, sigma=0.05, rank=4, seed=1) as n:
        assert n == 2  # both linears matched
        assert not torch.allclose(net(x), base)
    # Hooks removed ⇒ exact restoration, no bf16-style drift.
    assert torch.equal(net(x), base)


def test_distinct_seeds_give_distinct_members() -> None:
    net, x = _TinyNet(), torch.randn(4, 32)
    with perturbed(net, sigma=0.05, rank=4, seed=1):
        a = net(x).clone()
    with perturbed(net, sigma=0.05, rank=4, seed=2):
        b = net(x).clone()
    with perturbed(net, sigma=0.05, rank=4, seed=1):
        a2 = net(x).clone()
    assert not torch.allclose(a, b)
    assert torch.equal(a, a2)  # a member is reproducible from its seed


def test_relative_perturbation_size_matches_sigma() -> None:
    """‖ΔW‖_F / ‖W‖_F should track sigma — that is what makes a sweep readable."""
    torch.manual_seed(0)
    d, sigma = 128, 0.1
    net = _TinyNet(d)
    w = net.o_proj.weight
    eye = torch.eye(d)
    with torch.no_grad():
        base = torch.nn.functional.linear(eye, w)
        with perturbed(net, sigma=sigma, rank=16, seed=3):
            pert = net.o_proj(eye)
    ratio = float(torch.linalg.norm(pert - base) / torch.linalg.norm(w.detach()))
    # Random low-rank draw ⇒ concentration is loose at rank 16; assert the order.
    assert 0.5 * sigma < ratio < 2.0 * sigma


def test_no_matching_layers_raises() -> None:
    with pytest.raises(ValueError, match="no nn.Linear matched"):
        with perturbed(nn.Linear(4, 4), sigma=0.1):
            pass


def test_auroc_known_cases() -> None:
    assert auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auroc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert auroc([1.0], [1.0]) == 0.5  # ties count half
    assert np.isnan(auroc([], [1.0]))


def test_evaluate_perfect_and_matched_fpr() -> None:
    """A separable case: corrupt span always largest ⇒ catch 1.0, auroc 1.0."""
    planted = [
        {"scores": [1.0, 9.0, 2.0], "std": [0.0] * 3, "corrupt_i": 1}
        for _ in range(10)
    ]
    clean = [{"scores": [1.0, 2.0, 1.5], "std": [0.0] * 3, "corrupt_i": None}
             for _ in range(10)]
    m = _evaluate(planted, clean, clean, target_fpr=0.02)
    assert m["auroc"] == 1.0
    assert m["catch_rate"] == 1.0
    assert m["clean_fp_rate"] == 0.0
    assert m["n_planted"] == 10


def test_evaluate_floor_blocks_a_correct_argmax() -> None:
    """Argmax right but below the floor ⇒ not a catch (the F21e trade)."""
    planted = [{"scores": [1.0, 2.1], "std": [0.0, 0.0], "corrupt_i": 1}]
    calib = [{"scores": [1.0, 5.0], "std": [0.0, 0.0], "corrupt_i": None}
             for _ in range(10)]
    m = _evaluate(planted, calib, calib, target_fpr=0.02)
    assert m["catch_rate"] == 0.0  # floor ≈ 5.0 > 2.1


def test_evaluate_floor_comes_from_calib_not_eval() -> None:
    """The held-out split must drive the floor; swapping it must change results."""
    planted = [{"scores": [1.0, 6.0], "std": [0.0, 0.0], "corrupt_i": 1}]
    low = [{"scores": [1.0, 2.0], "std": [0.0, 0.0], "corrupt_i": None}
           for _ in range(10)]
    high = [{"scores": [1.0, 20.0], "std": [0.0, 0.0], "corrupt_i": None}
            for _ in range(10)]
    # Floor from the LOW calib set ⇒ the planted lie (6.0) clears it.
    assert _evaluate(planted, high, low, target_fpr=0.02)["catch_rate"] == 1.0
    # Floor from the HIGH calib set ⇒ it does not.
    assert _evaluate(planted, low, high, target_fpr=0.02)["catch_rate"] == 0.0
    # clean_fp_rate is measured on the EVAL split, so the two differ.
    assert _evaluate(planted, high, low, target_fpr=0.02)["clean_fp_rate"] == 1.0
    assert _evaluate(planted, low, high, target_fpr=0.02)["clean_fp_rate"] == 0.0


def test_arm_defaults() -> None:
    assert Arm("x", k=1, sigma=0.0).rank == 8
