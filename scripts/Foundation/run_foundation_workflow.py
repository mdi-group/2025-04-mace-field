#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "MACEField-omat-dielectric.model"
DEFAULT_OUTPUT_ROOT = HERE / "analysis_outputs" / "foundation_workflow"
DEFAULT_HESSIAN_INPUT = HERE.parent / "Dielectrics" / "MP-Dielectrics-filtered-valid.xyz"
REPLAY_DATASET_NAME = "replay_mh0_omat_pbe"
FALLBACK_HESSIAN_INPUTS = [
    HERE.parent / "Dielectrics" / "MP-Dielectrics-filtered-valid.xyz",
    HERE.parent / "Dielectrics" / "MP-Dielectrics-valid.xyz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the foundation parity/error workflow plus Hessian/ionic dielectric analysis."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--head", default="pt_head")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA ids. If multiple are given, parity uses the first and Hessians use the second.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--disable-cueq", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-hessian", action="store_true")
    parser.add_argument("--hessian-input", type=Path, default=DEFAULT_HESSIAN_INPUT)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--maxsteps", type=int, default=300)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--eigval-tol", type=float, default=1e-5)
    parser.add_argument("--asr-tol", type=float, default=1e-4)
    parser.add_argument("--bec-asr-tol", type=float, default=0.05)
    parser.add_argument("--principal", action="store_true")
    parser.add_argument("--debug-mass-weighted", action="store_true")
    parser.add_argument("--debug-n", type=int, default=5)
    return parser.parse_args()


def split_gpus(gpus: str | None) -> tuple[str | None, str | None]:
    if not gpus:
        return None, None
    parts = [part.strip() for part in gpus.split(",") if part.strip()]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_hessian_input(requested: Path) -> Path:
    if requested.exists():
        return requested
    for candidate in FALLBACK_HESSIAN_INPUTS:
        if candidate.exists():
            print(f"Using Hessian input {candidate} because requested path {requested} was not found.")
            return candidate
    raise FileNotFoundError(
        f"Hessian input not found: {requested}. Checked fallbacks: "
        + ", ".join(str(path) for path in FALLBACK_HESSIAN_INPUTS)
    )


def parity_outputs_complete(parity_out: Path) -> bool:
    summary_path = parity_out / "tables" / "foundation_summary.json"
    expected = [
        summary_path,
        parity_out / "tables" / "foundation_metrics.csv",
        parity_out / "plots" / "energy_per_atom_parity.png",
        parity_out / "plots" / "forces_parity.png",
        parity_out / "plots" / "stress_parity.png",
        parity_out / "plots" / "polarizability_parity.png",
        parity_out / "plots" / "bec_parity.png",
        parity_out / "plots" / "polarization_parity.png",
    ]
    if not all(path.exists() for path in expected):
        return False

    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("energy_force_stress_source_dataset") != REPLAY_DATASET_NAME:
        return False

    prediction_dir = parity_out / "predictions"
    replay_predictions = list(prediction_dir.glob(f"{REPLAY_DATASET_NAME}__*.extxyz"))
    return bool(replay_predictions)


def filtered_keep_count(filter_summary_csv: Path) -> int | None:
    if not filter_summary_csv.exists():
        return None
    try:
        with filter_summary_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "keep" not in reader.fieldnames:
                return None
            keep_count = 0
            for row in reader:
                value = str(row.get("keep", "")).strip().lower()
                if value in {"1", "true", "t", "yes"}:
                    keep_count += 1
            return keep_count
    except OSError:
        return None


def hessian_outputs_complete(hessian_root: Path) -> bool:
    filter_summary_csv = hessian_root / "plots" / "filter_summary.csv"
    base_expected = [
        hessian_root / "dielectric_hessians.xyz",
        hessian_root / "dielectric_hessians.h5",
        hessian_root / "dielectric_hessians_filtered.h5",
        hessian_root / "dielectric_ionic_eps.xyz",
        filter_summary_csv,
        hessian_root / "plots" / "unfiltered_ionic_dielectric_parity.png",
    ]
    if not all(path.exists() for path in base_expected):
        return False

    keep_count = filtered_keep_count(filter_summary_csv)
    if keep_count == 0:
        return True

    filtered_expected = [
        hessian_root / "dielectric_hessians_filtered.xyz",
        hessian_root / "dielectric_ionic_eps_filtered.xyz",
        hessian_root / "plots" / "filtered_ionic_dielectric_parity.png",
    ]
    return all(path.exists() for path in filtered_expected)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    hessian_input = resolve_hessian_input(args.hessian_input)

    parity_gpu, hessian_gpu = split_gpus(args.gpus)

    if not args.skip_parity:
        parity_out = args.output_root / "parity"
        if parity_outputs_complete(parity_out) and not args.force:
            print(f"Skipping parity stage because outputs already exist in {parity_out}")
        else:
            cmd = [
                sys.executable,
                str(HERE / "plot_foundation.py"),
                "--model-path",
                str(args.model_path),
                "--head",
                args.head,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--output-dir",
                str(parity_out),
            ]
            if args.force:
                cmd.append("--force")
            if args.disable_cueq:
                cmd.append("--disable-cueq")
            if parity_gpu is not None:
                cmd.extend(["--gpus", parity_gpu])
            run_cmd(cmd, HERE)

    if not args.skip_hessian:
        hessian_root = args.output_root / "hessian"
        hessian_root.mkdir(parents=True, exist_ok=True)
        output_xyz = hessian_root / "dielectric_hessians.xyz"
        output_h5 = hessian_root / "dielectric_hessians.h5"
        filtered_xyz = hessian_root / "dielectric_hessians_filtered.xyz"
        filtered_h5 = hessian_root / "dielectric_hessians_filtered.h5"
        ionic_xyz = hessian_root / "dielectric_ionic_eps.xyz"
        filtered_ionic_xyz = hessian_root / "dielectric_ionic_eps_filtered.xyz"

        if hessian_outputs_complete(hessian_root) and not args.force:
            print(f"Skipping Hessian stage because outputs already exist in {hessian_root}")
        else:
            cmd = [
                sys.executable,
                str(HERE / "ht_hessian_mace_mp_d3.py"),
                str(hessian_input),
                str(output_xyz),
                str(output_h5),
                "--model",
                str(args.model_path),
                "--head",
                args.head,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--plots-dir",
                str(hessian_root / "plots"),
                "--filtered-xyz",
                str(filtered_xyz),
                "--filtered-h5",
                str(filtered_h5),
                "--eigval-tol",
                str(args.eigval_tol),
                "--asr-tol",
                str(args.asr_tol),
                "--bec-asr-tol",
                str(args.bec_asr_tol),
                "--ionic-output-xyz",
                str(ionic_xyz),
                "--filtered-ionic-output-xyz",
                str(filtered_ionic_xyz),
                "--fmax",
                str(args.fmax),
                "--maxsteps",
                str(args.maxsteps),
                "--debug-n",
                str(args.debug_n),
            ]
            if args.relax:
                cmd.append("--relax")
            if args.max is not None:
                cmd.extend(["--max", str(args.max)])
            if args.disable_cueq:
                cmd.append("--disable-cueq")
            if args.principal:
                cmd.append("--principal")
            if args.debug_mass_weighted:
                cmd.append("--debug-mass-weighted")
            if hessian_gpu is not None:
                cmd.extend(["--gpus", hessian_gpu])
            run_cmd(cmd, HERE)


if __name__ == "__main__":
    main()
