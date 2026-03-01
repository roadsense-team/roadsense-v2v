#!/usr/bin/env python3
"""Step 1 validation for emulator_params_measured.json (Phase 6 prep work)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union


PathPart = Union[str, int]
Diff = Tuple[Tuple[PathPart, ...], Any, Any]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_diffs(old: Any, new: Any, path: Tuple[PathPart, ...] = ()) -> List[Diff]:
    diffs: List[Diff] = []
    if isinstance(old, dict) and isinstance(new, dict):
        keys = sorted(set(old.keys()) | set(new.keys()))
        for key in keys:
            if key not in old:
                diffs.append((path + (key,), None, new[key]))
            elif key not in new:
                diffs.append((path + (key,), old[key], None))
            else:
                diffs.extend(_flatten_diffs(old[key], new[key], path + (key,)))
        return diffs
    if isinstance(old, list) and isinstance(new, list):
        max_len = max(len(old), len(new))
        for idx in range(max_len):
            if idx >= len(old):
                diffs.append((path + (idx,), None, new[idx]))
            elif idx >= len(new):
                diffs.append((path + (idx,), old[idx], None))
            else:
                diffs.extend(_flatten_diffs(old[idx], new[idx], path + (idx,)))
        return diffs
    if old != new:
        diffs.append((path, old, new))
    return diffs


def _path_to_str(path: Sequence[PathPart]) -> str:
    out: List[str] = []
    for part in path:
        if isinstance(part, int):
            out.append(f"[{part}]")
        elif out:
            out.append(f".{part}")
        else:
            out.append(str(part))
    return "".join(out)


def _is_allowed_diff(path: Sequence[PathPart]) -> bool:
    if not path:
        return False
    if path[0] == "_metadata":
        return True
    return tuple(path) in {
        ("burst_loss", "mean_burst_length"),
        ("domain_randomization", "loss_rate_range", 1),
    }


def _approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Step 1 emulator parameter updates.")
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("espnow_emulator/emulator_params_measured.json"),
        help="Path to active measured params JSON.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=Path("espnow_emulator/emulator_params_pre_v3_backup.json"),
        help="Path to pre-v3 backup JSON.",
    )
    parser.add_argument(
        "--analysis-summary",
        type=Path,
        default=Path("data/convoy_analysis_site/analysis_summary.json"),
        help="Path to Recording #2 analysis summary JSON.",
    )
    args = parser.parse_args()

    params = _load_json(args.params)
    backup = _load_json(args.backup)
    rec2 = _load_json(args.analysis_summary)

    failures: List[str] = []

    burst = float(params["burst_loss"]["mean_burst_length"])
    if not (1.5 <= burst <= 1.7):
        failures.append(f"burst_loss.mean_burst_length expected ~1.6, got {burst:.6f}")

    dr_upper = float(params["domain_randomization"]["loss_rate_range"][1])
    if not (0.2316 <= dr_upper <= 0.30):
        failures.append(
            f"domain_randomization.loss_rate_range[1] expected in [0.2316, 0.30], got {dr_upper:.6f}"
        )

    metadata = params.get("_metadata", {})
    required_metadata = [
        "validated_against",
        "rec2_v002_v001_pdr",
        "rec2_v002_v001_latency_p95_ms",
        "validation_date",
    ]
    for key in required_metadata:
        if key not in metadata:
            failures.append(f"_metadata missing required key: {key}")

    rec2_pdr = float(rec2["pdr_by_link"]["V002->V001"]["pdr"])
    rec2_p95 = float(rec2["latency_by_link"]["V002->V001"]["relative_ms"]["p95"])
    rec2_burst = float(rec2["burst_by_link"]["V002->V001"]["mean_burst_length"])
    rec1_burst = float(backup["burst_loss"]["mean_burst_length"])
    expected_burst = (rec1_burst + rec2_burst) / 2.0

    if "rec2_v002_v001_pdr" in metadata and not _approx_equal(float(metadata["rec2_v002_v001_pdr"]), rec2_pdr):
        failures.append(
            f"_metadata.rec2_v002_v001_pdr mismatch: {metadata['rec2_v002_v001_pdr']} vs rec2 {rec2_pdr:.12f}"
        )
    if "rec2_v002_v001_latency_p95_ms" in metadata and not _approx_equal(
        float(metadata["rec2_v002_v001_latency_p95_ms"]), rec2_p95
    ):
        failures.append(
            "_metadata.rec2_v002_v001_latency_p95_ms mismatch: "
            f"{metadata['rec2_v002_v001_latency_p95_ms']} vs rec2 {rec2_p95:.3f}"
        )
    if not _approx_equal(burst, expected_burst, tol=1e-3):
        failures.append(
            f"burst_loss.mean_burst_length expected {expected_burst:.6f} "
            f"from average(rec1={rec1_burst:.6f}, rec2={rec2_burst:.6f}), got {burst:.6f}"
        )

    diffs = _flatten_diffs(backup, params)
    unexpected = [d for d in diffs if not _is_allowed_diff(d[0])]
    if unexpected:
        failures.append("unexpected non-Step1 changes detected against backup:")
        for path, old, new in unexpected[:10]:
            failures.append(f"  - {_path_to_str(path)}: {old!r} -> {new!r}")

    print("Step 1 validation summary")
    print(f"- burst_loss.mean_burst_length: {burst:.6f}")
    print(f"- domain_randomization.loss_rate_range[1]: {dr_upper:.6f}")
    print(f"- rec2 V002->V001 pdr: {rec2_pdr:.6f}")
    print(f"- rec2 V002->V001 latency p95 ms: {rec2_p95:.3f}")
    print(f"- rec2 V002->V001 mean burst: {rec2_burst:.6f}")
    print(f"- expected updated burst mean: {expected_burst:.6f}")

    if failures:
        print("\nFAILED:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print("\nPASS: Step 1 checks completed with no unexpected parameter changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
