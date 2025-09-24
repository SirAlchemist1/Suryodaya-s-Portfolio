#!/usr/bin/env python3
"""
tools/retarget_main_compat.py

A small compatibility wrapper that provides a `main()` entry point similar to a
project-level main, but internally uses the stable retarget pipeline you built
in tools/retarget_mmviolin.py.

Usage (CLI):

  python tools/retarget_main_compat.py \
    --input data/FullData/mmViolin_v2.0.npz \
    --mapping tools/mapping_example_18.json \
    --outdir out/mmViolin_v2_retargeted \
    --center_pelvis \
    --unit m \
    --nan_policy interp --interp_max_gap 5 \
    --summary_csv out/mmViolin_v2_retargeted/summary.csv \
    --dedupe last --min_frames 10
"""
from __future__ import annotations
import argparse
import json
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from retarget_mmviolin import (
    load_trials,
    load_mapping,
    save_outputs,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .npy or .npz (mmViolin_v2.0.npz)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--mapping", required=True, help="JSON mapping file (source->target)")

    # Dedupe + post-proc flags (match your retargeter)
    ap.add_argument("--save_individual", action="store_true", help="Save per-trial .npy outputs")
    ap.add_argument("--dedupe", choices=("first","last","error"), default="last",
                    help="Policy for duplicate PCT keys (default: last)")
    ap.add_argument("--center_pelvis", action="store_true",
                    help="Center on pelvis/root if present in target_names")
    ap.add_argument("--unit", choices=("mm","m"), default="mm",
                    help="Coordinate unit conversion (mm=identity, m=divide by 1000)")
    ap.add_argument("--nan_policy", choices=("keep","interp"), default="keep",
                    help="Keep NaNs or interpolate short gaps")
    ap.add_argument("--interp_max_gap", type=int, default=5,
                    help="Max consecutive NaN frames to interpolate")
    ap.add_argument("--min_frames", type=int, default=1,
                    help="Skip trials shorter than this length")
    ap.add_argument("--summary_csv", type=str, default=None,
                    help="Optional per-trial summary CSV")

    args = ap.parse_args()

    # Load mapping first to get an expected joint count for coercion
    try:
        with open(args.mapping, "r") as f:
            m = json.load(f)
        expected_joints: Optional[int] = len(m.get("source_names", [])) or None
    except Exception:
        expected_joints = None

    # Trials (examples or mmViolin)
    trials = load_trials(args.input, expected_joints=expected_joints)
    if not trials:
        raise RuntimeError("No trials found in input.")

    # Determine actual J from first trial, then (re)load mapping with identity fallback if needed
    J_src_hint = trials[0][1].shape[1]
    src_names, target_order, expanded_map, used_identity = load_mapping(args.mapping, J_src_hint)

    # Retarget & write
    save_outputs(
        outdir=args.outdir,
        trials=trials,
        mapping=(src_names, target_order, expanded_map, used_identity),
        save_individual=args.save_individual,
        dedupe=args.dedupe,
        center_pelvis=args.center_pelvis,
        unit=args.unit,
        nan_policy=args.nan_policy,
        interp_max_gap=args.interp_max_gap,
        min_frames=args.min_frames,
        summary_csv=args.summary_csv,
    )

if __name__ == "__main__":
    main()
