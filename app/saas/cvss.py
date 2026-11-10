"""CVSS v3.x base score calculator (FIRST CVSS v3.1 specification, section 7.1)."""

from __future__ import annotations

import math
from typing import Dict, Optional

_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_AC = {"L": 0.77, "H": 0.44}
_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}
_UI = {"N": 0.85, "R": 0.62}
_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}
_METRICS = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")


def _roundup(value: float) -> float:
    """Smallest number, to one decimal, that is >= value (spec Appendix A)."""
    as_int = round(value * 100000)
    if as_int % 10000 == 0:
        return as_int / 100000.0
    return (math.floor(as_int / 10000) + 1) / 10.0


def _parse_vector(vector: str) -> Optional[Dict[str, str]]:
    parts = vector.strip().split("/")
    if not parts or not parts[0].startswith("CVSS:3."):
        return None
    metrics: Dict[str, str] = {}
    for part in parts[1:]:
        key, sep, value = part.partition(":")
        if not sep:
            return None
        metrics[key] = value
    if any(m not in metrics for m in _METRICS):
        return None
    return metrics


def base_score(vector: str) -> Optional[float]:
    """CVSS v3.x base score for a vector string, or None when the vector is malformed."""
    m = _parse_vector(vector)
    if m is None:
        return None
    try:
        changed = m["S"] == "C"
        pr = (_PR_CHANGED if changed else _PR_UNCHANGED)[m["PR"]]
        iss = 1 - (1 - _CIA[m["C"]]) * (1 - _CIA[m["I"]]) * (1 - _CIA[m["A"]])
    except KeyError:
        return None
    if changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    if m["AV"] not in _AV or m["AC"] not in _AC or m["UI"] not in _UI or m["S"] not in ("U", "C"):
        return None
    exploitability = 8.22 * _AV[m["AV"]] * _AC[m["AC"]] * pr * _UI[m["UI"]]
    if impact <= 0:
        return 0.0
    if changed:
        return _roundup(min(1.08 * (impact + exploitability), 10.0))
    return _roundup(min(impact + exploitability, 10.0))


def severity_for_score(score: float) -> str:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"
