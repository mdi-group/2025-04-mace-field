#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from ase.io import read, write
from ase.stress import voigt_6_to_full_3x3_stress
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator


HERE = Path(__file__).resolve().parent
SCRIPTS_ROOT = HERE.parent
MACE_ROOT = Path.home() / "repositories" / "mace" / "mace-field"
if str(MACE_ROOT) not in sys.path:
    sys.path.insert(0, str(MACE_ROOT))

from mace.calculators import MACECalculator  # noqa: E402


DEFAULT_MODEL = HERE / "MACEField-omat-dielectric.model"
DEFAULT_OUTDIR = HERE / "analysis_outputs" / "foundation_model"
DEFAULT_HEAD = "pt_head"
PARITY_PLOT_DPI = 450
SCALAR_DENSITY_BINS = 180
TENSOR_DENSITY_BINS = 140
SCATTER_THRESHOLD = 20000
SCATTER_MAX_POINTS = 25000
REPLAY_DATASET_NAME = "replay_mh0_omat_pbe"
REPLAY_DATASET_PATH = HERE / "data" / "subselected-replay-data-mh-0-omat-pbe.xyz"

DATASETS = {
    REPLAY_DATASET_NAME: REPLAY_DATASET_PATH,
    "dielectric_unfiltered_train": SCRIPTS_ROOT / "Dielectrics" / "MP-Dielectrics-train.xyz",
    "dielectric_unfiltered_valid": SCRIPTS_ROOT / "Dielectrics" / "MP-Dielectrics-valid.xyz",
    "dielectric_unfiltered_test": SCRIPTS_ROOT / "Dielectrics" / "MP-Dielectrics-test.xyz",
    "dielectric_filtered_train": SCRIPTS_ROOT / "Dielectrics" / "MP-Dielectrics-filtered-train.xyz",
    "dielectric_filtered_valid": SCRIPTS_ROOT / "Dielectrics" / "MP-Dielectrics-filtered-valid.xyz",
    "dielectric_filtered_test": SCRIPTS_ROOT / "Dielectrics" / "MP-Dielectrics-filtered-test.xyz",
    "ferroelectric_train": SCRIPTS_ROOT / "Ferroelectrics" / "MP-Ferroelectrics-train.xyz",
    "ferroelectric_valid": SCRIPTS_ROOT / "Ferroelectrics" / "MP-Ferroelectrics-valid.xyz",
    "ferroelectric_test": SCRIPTS_ROOT / "Ferroelectrics" / "MP-Ferroelectrics-test.xyz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local foundation MACEField model and make parity/error plots."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--head", default=DEFAULT_HEAD)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA device ids to expose, e.g. '0' or '1,2'.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--force", action="store_true", help="Recompute cached predictions.")
    parser.add_argument(
        "--disable-cueq",
        action="store_true",
        help="Disable CuEq even on CUDA.",
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def apply_visible_gpus(gpus: str | None) -> None:
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def load_atoms(path: Path):
    return read(str(path), index=":")


def build_calculator(model_path: Path, head: str, device: str, dtype: str, enable_cueq: bool):
    return MACECalculator(
        model_paths=str(model_path),
        model_type="MACEField",
        default_dtype=dtype,
        device=device,
        head=head,
        enable_cueq=enable_cueq,
    )


def calculator_supported_atomic_numbers(calculator: MACECalculator) -> set[int]:
    z_table = getattr(calculator, "z_table", None)
    if z_table is not None and hasattr(z_table, "zs"):
        return {int(z) for z in z_table.zs}

    models = getattr(calculator, "models", None)
    if models:
        atomic_numbers = getattr(models[0], "atomic_numbers", None)
        if atomic_numbers is not None:
            return {int(z) for z in np.asarray(atomic_numbers, dtype=int).reshape(-1)}

    raise AttributeError("Could not determine supported atomic numbers from calculator")


def describe_atomic_numbers(atomic_numbers: list[int]) -> str:
    parts = []
    for z in atomic_numbers:
        symbol = chemical_symbols[z] if 0 < z < len(chemical_symbols) else f"Z{z}"
        parts.append(f"{symbol}({z})")
    return ", ".join(parts)


def evaluate_dataset(
    input_path: Path,
    output_path: Path,
    calculator: MACECalculator,
    head: str,
    force: bool,
):
    supported_atomic_numbers = calculator_supported_atomic_numbers(calculator)
    atoms_list = load_atoms(input_path)
    skipped_rows = []
    kept_atoms = []
    for idx, atoms in enumerate(atoms_list):
        unsupported = sorted(set(int(z) for z in atoms.numbers) - supported_atomic_numbers)
        if unsupported:
            skipped_rows.append(
                {
                    "dataset_input_path": str(input_path),
                    "frame_index": idx,
                    "material_id": atoms.info.get("material_id", ""),
                    "formula": atoms.get_chemical_formula(),
                    "unsupported_atomic_numbers": ";".join(str(z) for z in unsupported),
                    "unsupported_species": describe_atomic_numbers(unsupported),
                }
            )
            continue
        kept_atoms.append((idx, atoms))

    if skipped_rows:
        print(
            f"Skipping {len(skipped_rows)} unsupported structure(s) from {input_path.name}: "
            f"{', '.join(sorted({row['unsupported_species'] for row in skipped_rows}))}"
        )

    if output_path.exists() and not force:
        try:
            return load_atoms(output_path), skipped_rows
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Cached predictions at {output_path} could not be read ({type(exc).__name__}: {exc}); recomputing.")

    predicted = []
    for idx, atoms in kept_atoms:
        at = atoms.copy()
        at.info["head"] = head
        at.calc = calculator
        _ = at.get_potential_energy()
        results = dict(calculator.results)

        at.info["MACE_energy"] = float(np.asarray(results["energy"]).reshape(()))
        if "stress" in results and results["stress"] is not None:
            at.info["MACE_stress"] = stress_to_full_3x3(np.asarray(results["stress"], dtype=float)).reshape(9)
        if "polarization" in results and results["polarization"] is not None:
            at.info["MACE_polarization"] = np.asarray(results["polarization"], dtype=float).reshape(3)
        if "polarizability" in results and results["polarizability"] is not None:
            at.info["MACE_polarizability"] = np.asarray(results["polarizability"], dtype=float).reshape(9)
        if "forces" in results and results["forces"] is not None:
            at.arrays["MACE_forces"] = np.asarray(results["forces"], dtype=float)
        if "becs" in results and results["becs"] is not None:
            at.arrays["MACE_becs"] = np.asarray(results["becs"], dtype=float).reshape(len(at), 9)
        at.info["foundation_eval_idx"] = idx
        # Detach the shared live calculator before serialization so ASE does not
        # try to export stale per-atom results from a later frame.
        at.calc = None
        predicted.append(at)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), predicted)
    return predicted, skipped_rows


def stress_to_full_3x3(stress: np.ndarray) -> np.ndarray:
    arr = np.asarray(stress, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    if arr.size == 6:
        return np.asarray(voigt_6_to_full_3x3_stress(arr.reshape(6)), dtype=float)
    raise ValueError(f"Unsupported stress shape {arr.shape}")


def combine_datasets(*datasets):
    combined = []
    for dataset in datasets:
        combined.extend(dataset)
    return combined


def scalar_metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[mask]
    pred = pred[mask]
    resid = pred - ref
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    if ref.size >= 2:
        m, c = np.polyfit(ref, pred, 1)
        yhat = m * ref + c
        ss_res = float(np.sum((pred - yhat) ** 2))
        ss_tot = float(np.sum((pred - np.mean(pred)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    else:
        m, c, r2 = math.nan, math.nan, math.nan
    return {
        "n": int(ref.size),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": float(r2),
        "slope": float(m),
        "intercept": float(c),
    }


def plot_scalar_parity(ref, pred, title, units, outpath: Path, annotate: bool = True):
    ref = np.asarray(ref, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[mask]
    pred = pred[mask]
    metrics = scalar_metrics(ref, pred)

    vals = np.r_[ref, pred]
    lo, hi = np.percentile(vals, [1, 99])
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lim = (lo - pad, hi + pad)
    plot_mask = (
        np.isfinite(ref)
        & np.isfinite(pred)
        & (ref >= lim[0])
        & (ref <= lim[1])
        & (pred >= lim[0])
        & (pred <= lim[1])
    )
    plot_ref = ref[plot_mask]
    plot_pred = pred[plot_mask]

    fig, ax = plt.subplots(figsize=(5.6, 5.4), constrained_layout=True)
    if plot_ref.size <= SCATTER_THRESHOLD:
        if plot_ref.size > SCATTER_MAX_POINTS:
            sample_idx = np.linspace(0, plot_ref.size - 1, SCATTER_MAX_POINTS, dtype=int)
            plot_ref = plot_ref[sample_idx]
            plot_pred = plot_pred[sample_idx]
        ax.scatter(plot_ref, plot_pred, s=8, alpha=0.28, c="tab:blue", edgecolors="none")
        density_artist = None
    else:
        _, _, _, density_artist = ax.hist2d(
            plot_ref,
            plot_pred,
            bins=SCALAR_DENSITY_BINS,
            range=[lim, lim],
            norm=LogNorm(),
            cmin=1,
            cmap="viridis",
        )
    ax.plot(lim, lim, "--", color="0.3", lw=1.2)
    if np.isfinite(metrics["slope"]):
        xx = np.linspace(*lim, 200)
        ax.plot(xx, metrics["slope"] * xx + metrics["intercept"], color="tab:orange", lw=1.4)
    if annotate:
        txt = (
            f"R²={metrics['r2']:.3f}\n"
            f"RMSE={metrics['rmse']:.4g}\n"
            f"MAE={metrics['mae']:.4g}"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
        )
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Reference ({units})")
    ax.set_ylabel(f"Prediction ({units})")
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if density_artist is not None:
        cb = fig.colorbar(density_artist, ax=ax, pad=0.02)
        cb.set_label("log density")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=PARITY_PLOT_DPI)
    plt.close(fig)
    return metrics


def plot_vector_parity(ref, pred, title, units, outpath: Path):
    ref = np.asarray(ref, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return plot_scalar_parity(ref, pred, title, units, outpath)


def _tensor_limits(ref: np.ndarray, pred: np.ndarray, diag_nonneg: bool):
    ref = np.asarray(ref, dtype=float).reshape(-1, 3, 3)
    pred = np.asarray(pred, dtype=float).reshape(-1, 3, 3)
    diag_ref = np.concatenate([ref[:, i, i] for i in range(3)])
    diag_pred = np.concatenate([pred[:, i, i] for i in range(3)])
    off_pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    off_ref = np.concatenate([ref[:, i, j] for i, j in off_pairs])
    off_pred = np.concatenate([pred[:, i, j] for i, j in off_pairs])

    def clean(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return a[mask], b[mask]

    diag_ref, diag_pred = clean(diag_ref, diag_pred)
    off_ref, off_pred = clean(off_ref, off_pred)
    dq = np.percentile(np.r_[diag_ref, diag_pred], [1, 99])
    oq = np.percentile(np.r_[off_ref, off_pred], [1, 99])
    if diag_nonneg:
        diag_lim = (0.0, max(dq[1], 0.0) * 1.05 if dq[1] > 0 else 1.0)
    else:
        dmax = max(abs(dq[0]), abs(dq[1]))
        diag_lim = (-1.05 * dmax, 1.05 * dmax if dmax > 0 else 1.0)
    omax = max(abs(oq[0]), abs(oq[1]))
    off_lim = (-1.05 * omax, 1.05 * omax if omax > 0 else 1.0)
    return diag_lim, off_lim


def plot_tensor_two_panel(ref, pred, title, units, outpath: Path, diag_nonneg: bool):
    ref = np.asarray(ref, dtype=float).reshape(-1, 3, 3)
    pred = np.asarray(pred, dtype=float).reshape(-1, 3, 3)
    diag_lim, off_lim = _tensor_limits(ref, pred, diag_nonneg=diag_nonneg)
    diag_ref = np.concatenate([ref[:, i, i] for i in range(3)])
    diag_pred = np.concatenate([pred[:, i, i] for i in range(3)])
    off_pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    off_ref = np.concatenate([ref[:, i, j] for i, j in off_pairs])
    off_pred = np.concatenate([pred[:, i, j] for i, j in off_pairs])

    fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.2), constrained_layout=True)
    for ax, x, y, lim, panel in (
        (axs[0], diag_ref, diag_pred, diag_lim, "Diagonals"),
        (axs[1], off_ref, off_pred, off_lim, "Off-diagonals"),
    ):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        metrics = scalar_metrics(x, y)
        plot_mask = (x >= lim[0]) & (x <= lim[1]) & (y >= lim[0]) & (y <= lim[1])
        plot_x = x[plot_mask]
        plot_y = y[plot_mask]
        if plot_x.size <= SCATTER_THRESHOLD:
            if plot_x.size > SCATTER_MAX_POINTS:
                sample_idx = np.linspace(0, plot_x.size - 1, SCATTER_MAX_POINTS, dtype=int)
                plot_x = plot_x[sample_idx]
                plot_y = plot_y[sample_idx]
            ax.scatter(plot_x, plot_y, s=8, alpha=0.28, c="tab:blue", edgecolors="none")
            density_artist = None
        else:
            _, _, _, density_artist = ax.hist2d(
                plot_x,
                plot_y,
                bins=TENSOR_DENSITY_BINS,
                range=[lim, lim],
                norm=LogNorm(),
                cmin=1,
                cmap="viridis",
            )
        ax.plot(lim, lim, "--", color="0.3", lw=1.2)
        if np.isfinite(metrics["slope"]):
            xx = np.linspace(*lim, 200)
            ax.plot(xx, metrics["slope"] * xx + metrics["intercept"], color="tab:orange", lw=1.4)
        txt = f"R²={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.4g}\nMAE={metrics['mae']:.4g}"
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
        )
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(panel)
        ax.set_xlabel(f"Reference ({units})")
        ax.set_ylabel(f"Prediction ({units})")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if density_artist is not None:
            cb = fig.colorbar(density_artist, ax=ax, pad=0.02)
            cb.set_label("log density")
    fig.suptitle(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=PARITY_PLOT_DPI)
    plt.close(fig)
    return {
        "diag": scalar_metrics(diag_ref, diag_pred),
        "offdiag": scalar_metrics(off_ref, off_pred),
    }


def bec_asr_violation(becs: np.ndarray) -> float:
    arr = np.asarray(becs, dtype=float)
    if arr.shape[-1] == 9:
        arr = arr.reshape(-1, 3, 3)
    return float(np.max(np.abs(np.sum(arr, axis=0))))


def analyze_bec_asr(predicted_atoms, outdir: Path):
    columns = [
        "frame_index",
        "material_id",
        "formula",
        "n_atoms",
        "ref_bec_asr_max_abs",
        "pred_bec_asr_max_abs",
    ]
    rows = []
    for idx, at in enumerate(predicted_atoms):
        ref_err = bec_asr_violation(at.arrays["REF_becs"])
        pred_err = bec_asr_violation(at.arrays["MACE_becs"])
        rows.append(
            {
                "frame_index": idx,
                "material_id": at.info.get("material_id", ""),
                "formula": at.get_chemical_formula(),
                "n_atoms": len(at),
                "ref_bec_asr_max_abs": ref_err,
                "pred_bec_asr_max_abs": pred_err,
            }
        )
    df = pd.DataFrame(rows, columns=columns)
    csv_path = outdir / "bec_asr_summary.csv"
    if df.empty:
        df.to_csv(csv_path, index=False)
        print("No unfiltered dielectric predictions available for BEC ASR analysis; skipping histogram.")
        return df

    df = df.sort_values("ref_bec_asr_max_abs", ascending=False)
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    ref_counts, _, _ = ax.hist(df["ref_bec_asr_max_abs"], bins=80, alpha=0.6, label="REF")
    pred_counts, _, _ = ax.hist(df["pred_bec_asr_max_abs"], bins=80, alpha=0.6, label="MACE")
    if np.any(ref_counts > 0) or np.any(pred_counts > 0):
        ax.set_yscale("log")
    ax.set_xlabel(r"max$_{\alpha\beta} |\sum_\kappa Z^*_{\kappa,\alpha\beta}|$")
    ax.set_ylabel("Count")
    ax.set_title("BEC ASR deviation (unfiltered MP-Dielectrics)")
    ax.legend()
    fig.savefig(outdir / "bec_asr_histogram.png", dpi=250)
    plt.close(fig)
    return df


def collect_bec_component_rows(predicted_atoms):
    rows = []
    columns = [
        "material_id",
        "formula",
        "atom_index",
        "element",
        "component",
        "ref",
        "pred",
        "residual",
        "abs_residual",
    ]
    labels = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
    for at in predicted_atoms:
        ref = np.asarray(at.arrays["REF_becs"], dtype=float).reshape(len(at), 9)
        pred = np.asarray(at.arrays["MACE_becs"], dtype=float).reshape(len(at), 9)
        for atom_idx, symbol in enumerate(at.symbols):
            for comp_idx, comp in enumerate(labels):
                r = float(ref[atom_idx, comp_idx])
                p = float(pred[atom_idx, comp_idx])
                rows.append(
                    {
                        "material_id": at.info.get("material_id", ""),
                        "formula": at.get_chemical_formula(),
                        "atom_index": atom_idx,
                        "element": str(symbol),
                        "component": comp,
                        "ref": r,
                        "pred": p,
                        "residual": p - r,
                        "abs_residual": abs(p - r),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def analyze_bec_outliers(predicted_atoms, outdir: Path):
    df = collect_bec_component_rows(predicted_atoms)
    if df.empty:
        diag = pd.DataFrame(columns=df.columns)
        off = pd.DataFrame(columns=df.columns)
        summary = pd.DataFrame(columns=["kind", "element", "n", "median_abs_residual", "max_abs_residual"])
        diag.to_csv(outdir / "bec_diag_outliers.csv", index=False)
        off.to_csv(outdir / "bec_offdiag_outliers.csv", index=False)
        summary.to_csv(outdir / "bec_element_error_summary.csv", index=False)
        print("No unfiltered dielectric predictions available for BEC outlier analysis; skipping tables.")
        return diag, off, summary

    diag = df[df["component"].isin(["xx", "yy", "zz"])].sort_values("abs_residual", ascending=False)
    off = df[~df["component"].isin(["xx", "yy", "zz"])].sort_values("abs_residual", ascending=False)
    diag.to_csv(outdir / "bec_diag_outliers.csv", index=False)
    off.to_csv(outdir / "bec_offdiag_outliers.csv", index=False)
    summary = (
        pd.concat(
            [
                diag.assign(kind="diag"),
                off.assign(kind="offdiag"),
            ],
            ignore_index=True,
        )
        .groupby(["kind", "element"])
        .agg(n=("abs_residual", "size"), median_abs_residual=("abs_residual", "median"), max_abs_residual=("abs_residual", "max"))
        .reset_index()
        .sort_values(["kind", "n", "median_abs_residual"], ascending=[True, False, False])
    )
    summary.to_csv(outdir / "bec_element_error_summary.csv", index=False)
    return diag, off, summary


def dataset_metrics_row(dataset_name: str, quantity: str, metrics: dict[str, float | int]):
    row = {"dataset": dataset_name, "quantity": quantity}
    row.update(metrics)
    return row


def main():
    args = parse_args()
    apply_visible_gpus(args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    plot_dir = args.output_dir / "plots"
    table_dir = args.output_dir / "tables"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    calc = build_calculator(
        model_path=args.model_path,
        head=args.head,
        device=args.device,
        dtype=args.dtype,
        enable_cueq=(args.device.startswith("cuda") and not args.disable_cueq),
    )

    predicted = {}
    skipped_rows = []
    for name, path in DATASETS.items():
        output_path = prediction_dir / f"{name}__{args.model_path.stem}.extxyz"
        if not path.exists():
            if output_path.exists() and not args.force:
                try:
                    predicted[name] = load_atoms(output_path)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(
                        f"Cached predictions at {output_path} could not be read "
                        f"({type(exc).__name__}: {exc}); skipping {name} because source dataset {path} is missing."
                    )
            continue
        dataset_predicted, dataset_skipped = evaluate_dataset(path, output_path, calc, args.head, args.force)
        predicted[name] = dataset_predicted
        for row in dataset_skipped:
            skipped_rows.append({"dataset": name, **row})

    dielectric_filtered_eval = combine_datasets(
        predicted.get("dielectric_filtered_valid", []),
        predicted.get("dielectric_filtered_test", []),
    )
    dielectric_unfiltered_all = combine_datasets(
        predicted.get("dielectric_unfiltered_train", []),
        predicted.get("dielectric_unfiltered_valid", []),
        predicted.get("dielectric_unfiltered_test", []),
    )
    replay_eval = predicted.get(REPLAY_DATASET_NAME, [])
    ferroelectric_eval = combine_datasets(
        predicted.get("ferroelectric_valid", []),
        predicted.get("ferroelectric_test", []),
    )

    metrics_rows: list[dict[str, float | int | str]] = []

    if not replay_eval:
        raise RuntimeError(
            f"No replay-set predictions were available for {REPLAY_DATASET_PATH}. "
            "Energy, force, and stress parity plots now require this dataset."
        )

    ref_energy = np.array([at.info["REF_energy"] / len(at) for at in replay_eval], dtype=float)
    pred_energy = np.array([at.info["MACE_energy"] / len(at) for at in replay_eval], dtype=float)
    metrics_rows.append(
        dataset_metrics_row(
            REPLAY_DATASET_NAME,
            "energy_per_atom_eV",
            plot_scalar_parity(
                ref_energy,
                pred_energy,
                "Foundation replay energy parity",
                "eV/atom",
                plot_dir / "energy_per_atom_parity.png",
            ),
        )
    )

    ref_forces = np.concatenate([np.asarray(at.arrays["REF_forces"], dtype=float).reshape(-1, 3) for at in replay_eval], axis=0)
    pred_forces = np.concatenate([np.asarray(at.arrays["MACE_forces"], dtype=float).reshape(-1, 3) for at in replay_eval], axis=0)
    metrics_rows.append(
        dataset_metrics_row(
            REPLAY_DATASET_NAME,
            "forces_eV_per_A",
            plot_vector_parity(
                ref_forces,
                pred_forces,
                "Foundation replay force parity",
                "eV/Å",
                plot_dir / "forces_parity.png",
            ),
        )
    )

    ref_stress = np.stack([np.asarray(at.info["REF_stress"], dtype=float).reshape(3, 3) for at in replay_eval], axis=0)
    pred_stress = np.stack([np.asarray(at.info["MACE_stress"], dtype=float).reshape(3, 3) for at in replay_eval], axis=0)
    stress_metrics = plot_tensor_two_panel(
        ref_stress,
        pred_stress,
        "Foundation replay stress parity",
        r"eV/$\AA^3$",
        plot_dir / "stress_parity.png",
        diag_nonneg=False,
    )
    metrics_rows.append(dataset_metrics_row(REPLAY_DATASET_NAME, "stress_diag_eV_per_A3", stress_metrics["diag"]))
    metrics_rows.append(dataset_metrics_row(REPLAY_DATASET_NAME, "stress_offdiag_eV_per_A3", stress_metrics["offdiag"]))

    ref_alpha = np.stack([np.asarray(at.info["REF_polarizability"], dtype=float).reshape(3, 3) for at in dielectric_filtered_eval], axis=0)
    pred_alpha = np.stack([np.asarray(at.info["MACE_polarizability"], dtype=float).reshape(3, 3) for at in dielectric_filtered_eval], axis=0)
    alpha_metrics = plot_tensor_two_panel(
        ref_alpha,
        pred_alpha,
        "Foundation dielectric polarizability parity",
        r"$\varepsilon_\infty - I$",
        plot_dir / "polarizability_parity.png",
        diag_nonneg=True,
    )
    metrics_rows.append(dataset_metrics_row("dielectric_filtered_valid_test", "polarizability_diag", alpha_metrics["diag"]))
    metrics_rows.append(dataset_metrics_row("dielectric_filtered_valid_test", "polarizability_offdiag", alpha_metrics["offdiag"]))

    ref_becs = np.concatenate([np.asarray(at.arrays["REF_becs"], dtype=float).reshape(-1, 3, 3) for at in dielectric_filtered_eval], axis=0)
    pred_becs = np.concatenate([np.asarray(at.arrays["MACE_becs"], dtype=float).reshape(-1, 3, 3) for at in dielectric_filtered_eval], axis=0)
    bec_metrics = plot_tensor_two_panel(
        ref_becs,
        pred_becs,
        "Foundation dielectric BEC parity",
        "e",
        plot_dir / "bec_parity.png",
        diag_nonneg=False,
    )
    metrics_rows.append(dataset_metrics_row("dielectric_filtered_valid_test", "bec_diag_e", bec_metrics["diag"]))
    metrics_rows.append(dataset_metrics_row("dielectric_filtered_valid_test", "bec_offdiag_e", bec_metrics["offdiag"]))

    ref_pol = np.stack([np.asarray(at.info["REF_polarization"], dtype=float).reshape(3) for at in ferroelectric_eval], axis=0) * 1602.176634
    pred_pol = np.stack([np.asarray(at.info["MACE_polarization"], dtype=float).reshape(3) for at in ferroelectric_eval], axis=0) * 1602.176634
    metrics_rows.append(
        dataset_metrics_row(
            "ferroelectric_valid_test",
            "polarization_uC_per_cm2",
            plot_vector_parity(
                ref_pol,
                pred_pol,
                "Foundation ferroelectric polarization parity",
                "μC/cm²",
                plot_dir / "polarization_parity.png",
            ),
        )
    )

    bec_asr_df = analyze_bec_asr(dielectric_unfiltered_all, table_dir)
    diag_outliers, off_outliers, bec_summary = analyze_bec_outliers(dielectric_unfiltered_all, table_dir)

    with (table_dir / "foundation_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "quantity", "n", "rmse", "mae", "bias", "r2", "slope", "intercept"],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_report_path = table_dir / "skipped_unsupported_structures.csv"
    if skipped_df.empty:
        skipped_df = pd.DataFrame(
            columns=[
                "dataset",
                "dataset_input_path",
                "frame_index",
                "material_id",
                "formula",
                "unsupported_atomic_numbers",
                "unsupported_species",
            ]
        )
    skipped_df.to_csv(skipped_report_path, index=False)

    summary = {
        "model_path": str(args.model_path),
        "head": args.head,
        "device": args.device,
        "dtype": args.dtype,
        "energy_force_stress_source_dataset": REPLAY_DATASET_NAME,
        "energy_force_stress_source_path": str(REPLAY_DATASET_PATH),
        "prediction_dir": str(prediction_dir),
        "plots_dir": str(plot_dir),
        "tables_dir": str(table_dir),
        "skipped_unsupported_structures_csv": str(skipped_report_path),
        "skipped_unsupported_structure_count": int(len(skipped_rows)),
        "bec_asr_top_ref_material": bec_asr_df.iloc[0].to_dict() if not bec_asr_df.empty else None,
        "bec_diag_outlier_count": int(len(diag_outliers)),
        "bec_offdiag_outlier_count": int(len(off_outliers)),
        "bec_error_summary_rows": int(len(bec_summary)),
    }
    with (table_dir / "foundation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote predictions to {prediction_dir}")
    print(f"Wrote plots to {plot_dir}")
    print(f"Wrote metrics/error tables to {table_dir}")


if __name__ == "__main__":
    main()
