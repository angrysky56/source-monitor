"""
F14 power analysis: does the gate add repair beyond rehearsal alone?

Pairs base-drop vs surp-drop-hard BY SEED from results/results.jsonl.
Records are deduplicated keeping the LAST occurrence per (arm, seed), so
post--1e9 reruns of seeds 0-2 supersede the earlier leaky (-30) records.

Per corruption type and metric, reports the per-seed increment
(positive = gate advantage: lower bsi / higher d1), its mean +/- sd, and an
EXACT two-sided sign-flip permutation p-value on the mean (2^n patterns —
no distributional assumptions, no scipy).

    uv run python -m source_monitor.f14_power
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np

ARMS = ("base-drop", "surp-drop-hard")
TYPES = ("ghost", "misloc", "phantom")


def load_last(path: str = "results/results.jsonl") -> dict[tuple[str, int], dict]:
    last: dict[tuple[str, int], dict] = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        last[(r["arm"], r["seed"])] = r
    return last


def exact_sign_p(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p for mean(diffs) != 0."""
    obs = abs(diffs.mean())
    n = len(diffs)
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if abs((diffs * np.array(signs)).mean()) >= obs - 1e-12:
            hits += 1
    return hits / 2 ** n


def metric_value(rec: dict, typ: str, metric: str) -> float:
    if metric == "bsi":
        return float(rec[typ]["blindspot_idx"])
    return float(rec[f"{typ}_recovery"].get("1", float("nan")))


def main() -> None:
    last = load_last()
    seeds = sorted({s for (a, s) in last if a in ARMS
                    and all((arm, s) in last for arm in ARMS)})
    print(f"paired seeds (n={len(seeds)}): {seeds}")
    if len(seeds) < 4:
        print("WARNING: exact test has little power below n=4 (p floor 2/2^n).")

    print(f"\n{'type':8s} {'metric':6s} {'base-drop':>22s} {'surp-drop-hard':>22s} "
          f"{'increment':>16s} {'p':>7s}")
    pooled: list[float] = []
    for typ in TYPES:
        for metric in ("bsi", "d1"):
            b = np.array([metric_value(last[("base-drop", s)], typ, metric)
                          for s in seeds])
            g = np.array([metric_value(last[("surp-drop-hard", s)], typ, metric)
                          for s in seeds])
            inc = (b - g) if metric == "bsi" else (g - b)  # + = gate advantage
            p = exact_sign_p(inc)
            print(f"{typ:8s} {metric:6s} {b.mean():>9.3f} ±{b.std():.3f}     "
                  f"{g.mean():>9.3f} ±{g.std():.3f}     "
                  f"{inc.mean():>+8.3f} ±{inc.std():.3f} {p:>7.3f}")
            if metric == "bsi":
                pooled.extend(inc.tolist())
    pooled_a = np.array(pooled)
    print(f"\npooled bsi increment over types (secondary; types within a seed "
          f"are correlated): {pooled_a.mean():+.4f} ±{pooled_a.std():.4f}")
    print("\nper-seed bsi table (base-drop / surp-drop-hard):")
    for typ in TYPES:
        row = "  ".join(
            f"s{s}:{metric_value(last[('base-drop', s)], typ, 'bsi'):.3f}"
            f"/{metric_value(last[('surp-drop-hard', s)], typ, 'bsi'):.3f}"
            for s in seeds)
        print(f"  {typ:8s} {row}")
    print("\nreading: increment > 0 with p < .05 on misloc/phantom bsi would "
          "establish the gate's contribution beyond rehearsal (F14).")


if __name__ == "__main__":
    main()
