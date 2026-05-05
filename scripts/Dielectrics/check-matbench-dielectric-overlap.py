#!/usr/bin/env python3
"""
Cross-check structural overlap between a local MP dielectric dataset and the
official Matbench dielectric benchmark dataset.

This script compares a local `.extxyz`/`.xyz` file such as `MP-Dielectrics.extxyz`
against the official Matbench dielectric benchmark payload from
`https://ml.materialsproject.org/projects/matbench_dielectric.json.gz`.

Because the Matbench dataset only exposes `structure` and the target `n`
(refractive index), overlap is determined by exact-ish structural matching
within reduced-formula buckets using pymatgen's StructureMatcher.

Requirements:
  pip install ase pymatgen requests tqdm

Example:
  python check-matbench-dielectric-overlap.py \
    --mp-dataset MP-Dielectrics.extxyz

  python check-matbench-dielectric-overlap.py \
    --mp-dataset MP-Dielectrics-filtered.extxyz \
    --csv-out filtered-vs-matbench-overlap.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from ase.io import read as ase_read
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MATBENCH_DIELECTRIC_URL = "https://ml.materialsproject.org/projects/matbench_dielectric.json.gz"
DEFAULT_CACHE_NAME = "matbench_dielectric.json.gz"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Codex overlap checker)"
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
FOUNDATION_MODEL = SCRIPTS_ROOT / "Foundation" / "MACEField-omat-dielectric.model"
MACE_ROOT = Path.home() / "repositories" / "mace" / "mace-field"


def _resolve_output_path(base_path: Path, requested: str, suffix: str, ext: str) -> Path:
    if requested.strip():
        path = Path(requested)
        return path if path.is_absolute() else base_path.parent / path
    return base_path.with_name(f"{base_path.stem}{suffix}{ext}")


def _read_atoms_list(path: Path) -> List[Any]:
    atoms = ase_read(path, index=":")
    return atoms if isinstance(atoms, list) else [atoms]


def _load_local_mp_records(path: Path) -> List[Dict[str, Any]]:
    adaptor = AseAtomsAdaptor()
    atoms_list = _read_atoms_list(path)
    records: List[Dict[str, Any]] = []

    for idx, atoms in enumerate(atoms_list):
        structure = adaptor.get_structure(atoms)
        records.append(
            {
                "source_index": idx,
                "material_id": atoms.info.get("material_id"),
                "dielectric_task_id": atoms.info.get("dielectric_task_id"),
                "formula": structure.composition.reduced_formula,
                "nsites": len(structure),
                "structure": structure,
            }
        )
    return records


def _download_matbench_payload(cache_path: Path, refresh: bool, timeout: float) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        MATBENCH_DIELECTRIC_URL,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.content


def _load_matbench_records(cache_path: Path, refresh: bool, timeout: float) -> List[Dict[str, Any]]:
    payload = _download_matbench_payload(cache_path=cache_path, refresh=refresh, timeout=timeout)
    decoded = gzip.decompress(payload).decode("utf-8")
    raw = json.loads(decoded)

    if not isinstance(raw, dict) or set(raw.keys()) != {"index", "columns", "data"}:
        raise ValueError("Unexpected Matbench dielectric payload format.")

    if raw["columns"] != ["structure", "n"]:
        raise ValueError(f"Unexpected Matbench dielectric columns: {raw['columns']}")

    records: List[Dict[str, Any]] = []
    for benchmark_index, row in zip(raw["index"], raw["data"]):
        structure = Structure.from_dict(row[0])
        records.append(
            {
                "benchmark_index": int(benchmark_index),
                "matbench_n": float(row[1]),
                "formula": structure.composition.reduced_formula,
                "nsites": len(structure),
                "structure": structure,
            }
        )
    return records


def _bucket_by_formula(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["formula"])].append(record)
    return buckets


def _overlap_pairs_and_summary(
    mp_records: Sequence[Dict[str, Any]],
    matbench_records: Sequence[Dict[str, Any]],
    *,
    ltol: float,
    stol: float,
    angle_tol: float,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    mp_by_formula = _bucket_by_formula(mp_records)
    matbench_by_formula = _bucket_by_formula(matbench_records)
    common_formulas = sorted(set(mp_by_formula) & set(matbench_by_formula))

    overlap_pairs: List[Dict[str, Any]] = []
    overlap_group_counter: Counter[str] = Counter()
    mp_overlap_indices: set[int] = set()
    mp_overlap_material_ids: set[str] = set()
    matbench_overlap_indices: set[int] = set()
    n_overlap_groups = 0

    iterator: Iterable[str] = common_formulas
    if tqdm is not None:
        iterator = tqdm(common_formulas, desc="Match structures", unit="formula")

    for formula in iterator:
        combined_structures = []
        record_map: Dict[int, Tuple[str, Dict[str, Any]]] = {}

        for record in mp_by_formula[formula]:
            structure = record["structure"]
            combined_structures.append(structure)
            record_map[id(structure)] = ("mp", record)

        for record in matbench_by_formula[formula]:
            structure = record["structure"]
            combined_structures.append(structure)
            record_map[id(structure)] = ("matbench", record)

        grouped = matcher.group_structures(combined_structures)
        for group in grouped:
            mp_group = [record_map[id(structure)][1] for structure in group if record_map[id(structure)][0] == "mp"]
            matbench_group = [
                record_map[id(structure)][1] for structure in group if record_map[id(structure)][0] == "matbench"
            ]
            if not mp_group or not matbench_group:
                continue

            n_overlap_groups += 1
            overlap_group_counter[formula] += 1

            for mp_record in mp_group:
                mp_overlap_indices.add(int(mp_record["source_index"]))
                material_id = mp_record.get("material_id")
                if material_id is not None:
                    mp_overlap_material_ids.add(str(material_id))

            for matbench_record in matbench_group:
                matbench_overlap_indices.add(int(matbench_record["benchmark_index"]))

            for mp_record in mp_group:
                for matbench_record in matbench_group:
                    overlap_pairs.append(
                        {
                            "formula": formula,
                            "mp_source_index": int(mp_record["source_index"]),
                            "mp_material_id": mp_record.get("material_id"),
                            "mp_dielectric_task_id": mp_record.get("dielectric_task_id"),
                            "mp_nsites": int(mp_record["nsites"]),
                            "matbench_index": int(matbench_record["benchmark_index"]),
                            "matbench_n": float(matbench_record["matbench_n"]),
                            "matbench_nsites": int(matbench_record["nsites"]),
                        }
                    )

    summary = {
        "matbench_url": MATBENCH_DIELECTRIC_URL,
        "n_mp_records": len(mp_records),
        "n_mp_unique_material_ids": len({str(r['material_id']) for r in mp_records if r.get("material_id") is not None}),
        "n_matbench_records": len(matbench_records),
        "n_common_formula_buckets": len(common_formulas),
        "n_overlap_groups": n_overlap_groups,
        "n_overlap_pairs": len(overlap_pairs),
        "n_mp_overlap_records": len(mp_overlap_indices),
        "n_mp_overlap_material_ids": len(mp_overlap_material_ids),
        "n_matbench_overlap_records": len(matbench_overlap_indices),
        "fraction_mp_overlap_records": (len(mp_overlap_indices) / len(mp_records)) if mp_records else 0.0,
        "fraction_matbench_overlap_records": (len(matbench_overlap_indices) / len(matbench_records)) if matbench_records else 0.0,
        "matcher": {
            "ltol": ltol,
            "stol": stol,
            "angle_tol": angle_tol,
        },
        "top_overlap_formulas": [
            {"formula": formula, "n_overlap_groups": count}
            for formula, count in overlap_group_counter.most_common(20)
        ],
    }

    if verbose:
        summary["overlap_formula_counts_full"] = dict(sorted(overlap_group_counter.items()))

    return overlap_pairs, summary


def _write_pairs_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "formula",
        "mp_source_index",
        "mp_material_id",
        "mp_dielectric_task_id",
        "mp_nsites",
        "matbench_index",
        "matbench_n",
        "matbench_nsites",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_calculator(model_path: Path, head: str, device: str, dtype: str, enable_cueq: bool):
    import sys

    if str(MACE_ROOT) not in sys.path:
        sys.path.insert(0, str(MACE_ROOT))
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=str(model_path),
        model_type="MACEField",
        default_dtype=dtype,
        device=device,
        head=head,
        enable_cueq=enable_cueq,
    )


def _structure_atomic_numbers(structure: Structure) -> List[int]:
    return sorted({int(sp.Z) for sp in structure.species})


def _predict_refractive_index(
    matbench_records: Sequence[Dict[str, Any]],
    *,
    model_path: Path,
    head: str,
    device: str,
    dtype: str,
    enable_cueq: bool,
    limit: int | None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    adaptor = AseAtomsAdaptor()
    calc = _build_calculator(
        model_path=model_path,
        head=head,
        device=device,
        dtype=dtype,
        enable_cueq=enable_cueq,
    )
    supported_zs = {int(z) for z in calc.z_table.zs}
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    records = list(matbench_records[:limit] if limit is not None else matbench_records)
    supported_records = []
    for record in records:
        atomic_numbers = _structure_atomic_numbers(record["structure"])
        unsupported = [z for z in atomic_numbers if z not in supported_zs]
        if unsupported:
            skipped.append(
                {
                    "matbench_index": int(record["benchmark_index"]),
                    "formula": str(record["formula"]),
                    "unsupported_atomic_numbers": unsupported,
                }
            )
            continue
        supported_records.append(record)

    iterator: Iterable[Dict[str, Any]] = supported_records
    if tqdm is not None:
        iterator = tqdm(supported_records, desc="Predict Matbench n", unit="structure")

    for record in iterator:
        atoms = adaptor.get_atoms(record["structure"])
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        results = dict(calc.results)
        alpha = np.asarray(results["polarizability"], dtype=float).reshape(3, 3)
        eps_inf_diag = 1.0 + np.diag(alpha)
        eps_inf_scalar = float(np.mean(eps_inf_diag))
        n_pred = float(np.sqrt(max(eps_inf_scalar, 0.0)))
        rows.append(
            {
                "matbench_index": int(record["benchmark_index"]),
                "formula": str(record["formula"]),
                "nsites": int(record["nsites"]),
                "reference_n": float(record["matbench_n"]),
                "predicted_n": n_pred,
                "predicted_eps_inf_scalar": eps_inf_scalar,
                "predicted_eps_inf_x": float(eps_inf_diag[0]),
                "predicted_eps_inf_y": float(eps_inf_diag[1]),
                "predicted_eps_inf_z": float(eps_inf_diag[2]),
            }
        )
    skip_summary = {
        "n_requested": len(records),
        "n_supported": len(supported_records),
        "n_skipped_unsupported": len(skipped),
        "supported_atomic_numbers": sorted(supported_zs),
        "skipped_examples": skipped[:50],
    }
    return rows, skip_summary


def _scalar_metrics(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ref = np.asarray(ref, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[mask]
    pred = pred[mask]
    resid = pred - ref
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
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
        "r2": float(r2),
        "slope": float(m),
        "intercept": float(c),
    }


def _write_prediction_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "matbench_index",
        "formula",
        "nsites",
        "reference_n",
        "predicted_n",
        "predicted_eps_inf_scalar",
        "predicted_eps_inf_x",
        "predicted_eps_inf_y",
        "predicted_eps_inf_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_prediction_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            clean = dict(row)
            for key in (
                "matbench_index",
                "nsites",
                "reference_n",
                "predicted_n",
                "predicted_eps_inf_scalar",
                "predicted_eps_inf_x",
                "predicted_eps_inf_y",
                "predicted_eps_inf_z",
            ):
                if key not in clean or clean[key] == "":
                    continue
                if key in {"matbench_index", "nsites"}:
                    clean[key] = int(float(clean[key]))
                else:
                    clean[key] = float(clean[key])
            rows.append(clean)
    return rows


def _point_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> np.ndarray:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    xi = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    density = hist[xi, yi]
    density[density < 1] = 1
    return density


def _plot_refractive_index_parity(path: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    title_fs = 20
    label_fs = 18
    tick_fs = 14
    inset_tick_fs = 12
    stats_fs = 17
    cbar_label_fs = 18
    cbar_tick_fs = 14

    ref = np.asarray([row["reference_n"] for row in rows], dtype=float)
    pred = np.asarray([row["predicted_n"] for row in rows], dtype=float)
    metrics = _scalar_metrics(ref, pred)

    full_max = float(np.max(np.r_[ref, pred]))
    full_lim = (-0.02 * max(full_max, 1.0), full_max * 1.03)

    q99 = float(np.percentile(np.r_[ref, pred], 99.0))
    inset_lim = (1.0, q99 * 1.03)

    full_density = _point_density(ref, pred, bins=120, xlim=full_lim, ylim=full_lim)
    inset_density = _point_density(ref, pred, bins=90, xlim=inset_lim, ylim=inset_lim)
    vmax = max(float(np.max(full_density)), float(np.max(inset_density)), 1.0)
    norm = LogNorm(vmin=1.0, vmax=vmax)

    order_full = np.argsort(full_density)
    order_inset = np.argsort(inset_density)

    fig, ax = plt.subplots(figsize=(8.6, 7.6), constrained_layout=True)
    sc = ax.scatter(
        ref[order_full],
        pred[order_full],
        c=full_density[order_full],
        s=14,
        cmap="viridis",
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    ax.plot(full_lim, full_lim, "--", color="0.4", lw=1.2)
    if np.isfinite(metrics["slope"]):
        xx = np.linspace(*full_lim, 200)
        ax.plot(xx, metrics["slope"] * xx + metrics["intercept"], color="tab:orange", lw=1.5)
    txt = (
        f"$R^2={metrics['r2']:.3f}$\n"
        f"RMSE={metrics['rmse']:.3f}\n"
        f"MAE={metrics['mae']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=stats_fs,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
    )
    ax.set_xlim(*full_lim)
    ax.set_ylim(*full_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Reference refractive index (unitless)", fontsize=label_fs)
    ax.set_ylabel("MACE refractive index (unitless)", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs, width=1.6, length=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(1.6)

    axins = inset_axes(
        ax,
        width="54%",
        height="54%",
        loc="upper left",
        bbox_to_anchor=(0.40, -0.03, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.8,
    )
    axins.scatter(
        ref[order_inset],
        pred[order_inset],
        c=inset_density[order_inset],
        s=10,
        cmap="viridis",
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    axins.plot(inset_lim, inset_lim, "--", color="0.4", lw=1.0)
    if np.isfinite(metrics["slope"]):
        xx_in = np.linspace(*inset_lim, 200)
        axins.plot(xx_in, metrics["slope"] * xx_in + metrics["intercept"], color="tab:orange", lw=1.2)
    axins.set_xlim(*inset_lim)
    axins.set_ylim(*inset_lim)
    axins.set_aspect("equal", adjustable="box")
    axins.set_box_aspect(1)
    axins.tick_params(axis="both", labelsize=inset_tick_fs, width=1.2, length=4)
    for spine in axins.spines.values():
        spine.set_linewidth(1.2)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.55", ls="--", lw=0.9)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log density", fontsize=cbar_label_fs)
    cb.ax.tick_params(labelsize=cbar_tick_fs, width=1.4, length=5)
    fig.suptitle("Refractive index parity", fontsize=title_fs, y=0.985)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mp-dataset",
        default="MP-Dielectrics.extxyz",
        help="Local MP dielectric dataset to compare against Matbench dielectric.",
    )
    parser.add_argument(
        "--matbench-cache",
        default=DEFAULT_CACHE_NAME,
        help="Path to cache the official Matbench dielectric json.gz payload.",
    )
    parser.add_argument(
        "--refresh-matbench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-download the Matbench dielectric payload even if a cache file exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for downloading the Matbench dielectric payload.",
    )
    parser.add_argument("--matcher-ltol", type=float, default=0.2, help="StructureMatcher lattice tolerance.")
    parser.add_argument("--matcher-stol", type=float, default=0.3, help="StructureMatcher site tolerance.")
    parser.add_argument("--matcher-angle-tol", type=float, default=5.0, help="StructureMatcher angle tolerance in degrees.")
    parser.add_argument(
        "--csv-out",
        default="",
        help="CSV path for matched MP/Matbench pairs. Defaults to '<mp-dataset>-vs-matbench-dielectric-overlap.csv'.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="JSON path for the summary report. Defaults to '<mp-dataset>-vs-matbench-dielectric-overlap.json'.",
    )
    parser.add_argument(
        "--predict-refractive-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run the local foundation model on the Matbench dielectric structures and make an n parity plot.",
    )
    parser.add_argument(
        "--model-path",
        default=str(FOUNDATION_MODEL),
        help="Path to the local MACEField dielectric model.",
    )
    parser.add_argument(
        "--head",
        default="pt_head",
        help="Model head to use for prediction.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
        help="Torch device for model inference.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Calculator dtype.",
    )
    parser.add_argument(
        "--disable-cueq",
        action="store_true",
        help="Disable CuEq even on CUDA.",
    )
    parser.add_argument(
        "--prediction-csv-out",
        default="",
        help="CSV path for Matbench refractive-index predictions. Defaults to '<mp-dataset>-matbench-refractive-index-predictions.csv'.",
    )
    parser.add_argument(
        "--prediction-json-out",
        default="",
        help="JSON path for Matbench refractive-index prediction summary. Defaults to '<mp-dataset>-matbench-refractive-index-summary.json'.",
    )
    parser.add_argument(
        "--prediction-plot-out",
        default="",
        help="PNG path for the refractive-index parity plot. Defaults to '<mp-dataset>-matbench-refractive-index-parity.png'.",
    )
    parser.add_argument(
        "--prediction-limit",
        type=int,
        default=None,
        help="Optional limit on the number of Matbench structures to predict, useful for smoke tests.",
    )
    parser.add_argument(
        "--reuse-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing prediction CSV instead of rerunning model inference.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mp_dataset_path = Path(args.mp_dataset)
    if not mp_dataset_path.is_absolute():
        mp_dataset_path = SCRIPT_DIR / mp_dataset_path
    if not mp_dataset_path.exists():
        raise SystemExit(f"MP dielectric dataset not found: {mp_dataset_path}")

    cache_path = Path(args.matbench_cache)
    if not cache_path.is_absolute():
        cache_path = mp_dataset_path.parent / cache_path

    csv_out = _resolve_output_path(mp_dataset_path, args.csv_out, "-vs-matbench-dielectric-overlap", ".csv")
    json_out = _resolve_output_path(mp_dataset_path, args.json_out, "-vs-matbench-dielectric-overlap", ".json")
    prediction_csv_out = _resolve_output_path(
        mp_dataset_path,
        args.prediction_csv_out,
        "-matbench-refractive-index-predictions",
        ".csv",
    )
    prediction_json_out = _resolve_output_path(
        mp_dataset_path,
        args.prediction_json_out,
        "-matbench-refractive-index-summary",
        ".json",
    )
    prediction_plot_out = _resolve_output_path(
        mp_dataset_path,
        args.prediction_plot_out,
        "-matbench-refractive-index-parity",
        ".png",
    )

    mp_records = _load_local_mp_records(mp_dataset_path)
    matbench_records = _load_matbench_records(
        cache_path=cache_path,
        refresh=args.refresh_matbench,
        timeout=args.timeout,
    )
    overlap_pairs, summary = _overlap_pairs_and_summary(
        mp_records=mp_records,
        matbench_records=matbench_records,
        ltol=args.matcher_ltol,
        stol=args.matcher_stol,
        angle_tol=args.matcher_angle_tol,
        verbose=args.verbose,
    )

    summary["mp_dataset"] = str(mp_dataset_path)
    summary["matbench_cache"] = str(cache_path)
    summary["csv_out"] = str(csv_out)
    summary["json_out"] = str(json_out)

    _write_pairs_csv(csv_out, overlap_pairs)
    _write_summary_json(json_out, summary)

    if args.predict_refractive_index:
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")
        if args.reuse_predictions and prediction_csv_out.exists():
            prediction_rows = _read_prediction_csv(prediction_csv_out)
            prediction_skip_summary = {
                "n_requested": len(prediction_rows),
                "n_supported": len(prediction_rows),
                "n_skipped_unsupported": None,
                "supported_atomic_numbers": None,
                "skipped_examples": [],
                "cache_reused": True,
            }
        else:
            prediction_rows, prediction_skip_summary = _predict_refractive_index(
                matbench_records,
                model_path=model_path,
                head=args.head,
                device=args.device,
                dtype=args.dtype,
                enable_cueq=(not args.disable_cueq and args.device.startswith("cuda")),
                limit=args.prediction_limit,
            )
            _write_prediction_csv(prediction_csv_out, prediction_rows)
        prediction_metrics = _plot_refractive_index_parity(prediction_plot_out, prediction_rows)
        prediction_summary = {
            "model_path": str(model_path),
            "head": args.head,
            "device": args.device,
            "dtype": args.dtype,
            "n_structures": len(prediction_rows),
            "definition": "predicted_n = sqrt(mean(diag(1 + polarizability_tensor)))",
            "metrics": prediction_metrics,
            "prediction_csv": str(prediction_csv_out),
            "prediction_plot": str(prediction_plot_out),
            "prediction_limit": args.prediction_limit,
            "skip_summary": prediction_skip_summary,
        }
        _write_summary_json(prediction_json_out, prediction_summary)
        summary["prediction_csv_out"] = str(prediction_csv_out)
        summary["prediction_json_out"] = str(prediction_json_out)
        summary["prediction_plot_out"] = str(prediction_plot_out)
        summary["prediction_metrics"] = prediction_metrics
        summary["prediction_skip_summary"] = prediction_skip_summary
        _write_summary_json(json_out, summary)

    print(f"MP dataset:          {mp_dataset_path}")
    print(f"Matbench cache:      {cache_path}")
    print(f"Matched pair rows:   {summary['n_overlap_pairs']}")
    print(f"Overlap groups:      {summary['n_overlap_groups']}")
    print(
        "MP overlaps:         "
        f"{summary['n_mp_overlap_records']} / {summary['n_mp_records']} frames "
        f"({summary['fraction_mp_overlap_records']:.1%})"
    )
    print(
        "Matbench overlaps:   "
        f"{summary['n_matbench_overlap_records']} / {summary['n_matbench_records']} rows "
        f"({summary['fraction_matbench_overlap_records']:.1%})"
    )
    print(f"Pair CSV:            {csv_out}")
    print(f"Summary JSON:        {json_out}")
    if args.predict_refractive_index:
        print(f"Prediction CSV:      {prediction_csv_out}")
        print(f"Prediction JSON:     {prediction_json_out}")
        print(f"Parity plot:         {prediction_plot_out}")
        print(
            "Predicted subset:    "
            f"{summary['prediction_skip_summary']['n_supported']} / "
            f"{summary['prediction_skip_summary']['n_requested']} supported by model"
        )


if __name__ == "__main__":
    main()
