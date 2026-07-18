"""
source-monitor — source monitoring as an architectural mechanism.

Provenance-tagged tokens + a learned, depth-causal admission gate that can
discount self-emitted evidence in attention. Tests whether the self-correction
blind spot (an echo / backtracking-inference phenomenon) is reduced by giving
the model an explicit self/external distinction and a mechanism to act on it.

Vendored (copied, not path-imported) from `sps-blindspot`: task, Muon, and the
ghost/Jacobian instruments. Projects stay self-contained.
"""

__version__ = "0.1.0"
