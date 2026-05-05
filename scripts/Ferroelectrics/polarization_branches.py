#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.io import read, write
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogFormatter, ScalarFormatter
from tqdm.auto import tqdm


POL_UC_CM2 = 1602.176634
DEFAULT_DTYPE = "float64"
DEFAULT_ENABLE_CUEQ = True
DEFAULT_PLOT_DPI = 300
PLOT_FONTSIZE = 15
LEGACY_MODEL_TYPES = {"ScaleShiftFieldMACE"}
DEFAULT_CODE_ROOTS = {
    "ScaleShiftFieldMACE": Path.home() / "repositories" / "mace-preprint" / "mace-field",
    "MACEField": Path.home() / "repositories" / "mace" / "mace-field",
}
DEFAULT_MODEL_NAMES = (
    "MACE-Field-MP-Ferroelectrics.model",
    "MACEField-omat-dielectric.model",
)

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.size": PLOT_FONTSIZE,
        "axes.labelsize": PLOT_FONTSIZE,
        "axes.titlesize": PLOT_FONTSIZE,
        "xtick.labelsize": PLOT_FONTSIZE,
        "ytick.labelsize": PLOT_FONTSIZE,
        "legend.fontsize": 10,
        "figure.labelsize": PLOT_FONTSIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def set_axis_fontsizes(ax, fontsize: int = PLOT_FONTSIZE) -> None:
    """Apply a consistent fontsize to axis labels, tick labels, and titles."""
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize)


@dataclass
class RuntimeConfig:
    root: Path
    analysis_dir: Path
    figure_dir: Path
    dataset_paths: dict[str, Path]
    material_indices: list[int]
    force_rerun: bool
    default_dtype: str
    electric_field: tuple[float, float, float] | None
    preferred_device: str | None
    legacy_device: str | None
    gpu_index: int | None
    enable_cueq: bool
    cueq_override: bool | None
    head_override: str | None
    model_type_override: str | None
    code_root_override: Path | None
    plot_dpi: int


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    default_output_dir = script_root / "analysis_outputs" / "polarization_branches"

    parser = argparse.ArgumentParser(
        description=(
            "Convert the polarization_branches notebook into a standalone analysis "
            "script that evaluates local ferroelectrics models, caches predictions, "
            "and writes plots plus tabular outputs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_root,
        help="Ferroelectrics script directory containing the MP-Ferroelectrics *.xyz files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Model paths to analyse. Relative paths are resolved against --root. "
            "Defaults to the two local checkpoints used in the notebook."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for plots and CSV/NPZ outputs.",
    )
    parser.add_argument(
        "--material-indices",
        nargs="+",
        type=int,
        default=[0],
        help="Path indices for example branch/component plots.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached predicted extxyz files and recompute model predictions.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device for non-legacy models. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--legacy-device",
        default=None,
        help="Override device for legacy ScaleShiftFieldMACE models. Defaults to cuda.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "CUDA GPU index to use, for example `--gpu 1` for `cuda:1`. "
            "Applied unless --device/--legacy-device already specify an explicit device."
        ),
    )
    parser.add_argument(
        "--head",
        default="mp-ferroelectric",
        help="Override MACE calculator head. Defaults to pt_head for MACEField and Default for legacy.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Override calculator model_type passed to MACECalculator.",
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="Override the MACE code checkout used to load the model class and calculator.",
    )
    parser.add_argument(
        "--default-dtype",
        default=DEFAULT_DTYPE,
        choices=("float32", "float64"),
        help="Default torch dtype for the calculator.",
    )
    parser.add_argument(
        "--enable-cueq",
        dest="cueq_override",
        action="store_true",
        default=None,
        help="Force CuEq on.",
    )
    parser.add_argument(
        "--disable-cueq",
        dest="cueq_override",
        action="store_false",
        help="Force CuEq off.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=DEFAULT_PLOT_DPI,
        help="DPI used when saving figure files.",
    )
    return parser.parse_args()


def resolve_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    required = [
        root / "MP-Ferroelectrics.xyz",
        root / "MP-Ferroelectrics-train.xyz",
        root / "MP-Ferroelectrics-valid.xyz",
        root / "MP-Ferroelectrics-test.xyz",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files under {root}: {', '.join(missing)}")
    return root


def resolve_model_inputs(root: Path, model_inputs: list[Path] | None) -> list[Path]:
    if model_inputs is None:
        model_inputs = [Path(name) for name in DEFAULT_MODEL_NAMES]

    resolved: list[Path] = []
    for model_input in model_inputs:
        model_path = model_input.expanduser()
        if not model_path.is_absolute():
            model_path = root / model_path
        model_path = model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        resolved.append(model_path)
    return resolved


def detect_model_family_from_file(model_path: Path) -> str:
    data = model_path.read_bytes()
    if b"ScaleShiftFieldMACE" in data:
        return "ScaleShiftFieldMACE"
    if b"MACEField" in data:
        return "MACEField"
    raise RuntimeError(
        f"Could not determine model family from {model_path}. "
        "Use --code-root if this is a non-standard checkpoint."
    )


def resolve_code_root(model_family: str, override: Path | None) -> Path:
    if override is not None:
        code_root = override.expanduser().resolve()
    else:
        code_root = DEFAULT_CODE_ROOTS[model_family].resolve()
    if not code_root.exists():
        raise FileNotFoundError(
            f"Expected code root for {model_family} at {code_root}. "
            "Use --code-root to override it."
        )
    return code_root


def load_mace_api(code_root: Path):
    code_root = code_root.resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    for module_name in list(sys.modules):
        if module_name == "mace" or module_name.startswith("mace."):
            del sys.modules[module_name]

    from mace.calculators import MACECalculator  # type: ignore

    try:
        from mace.tools import fold_polarization  # type: ignore
    except ImportError:
        from mace.modules.loss import fold_polarisation as _fold_polarisation  # type: ignore

        def fold_polarization(pred_polarization, ref_polarization, cell):
            delta = _fold_polarisation(pred_polarization, ref_polarization, cell)
            B = cell.view(-1, 3, 3)
            vol = torch.linalg.det(B).abs().clamp_min(1e-30).view(-1, 1, 1)
            qpol = B / vol
            fractional = torch.linalg.solve(
                qpol.transpose(-2, -1), delta.view(-1, 3).unsqueeze(-1)
            ).squeeze(-1)
            return delta, fractional

    return MACECalculator, fold_polarization


def inspect_model_file(model_path: Path) -> dict[str, str]:
    try:
        model = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_path.name} to inspect its class. "
            "Check that the matching MACE checkout is on PYTHONPATH."
        ) from exc

    model_class = model.__class__.__name__
    del model
    model_type = "ScaleShiftFieldMACE" if model_class == "ScaleShiftFieldMACE" else "MACEField"
    return {"class": model_class, "model_type": model_type}


def apply_gpu_index(device: str, gpu_index: int | None) -> str:
    if gpu_index is None:
        return device
    if device == "cuda":
        return f"cuda:{gpu_index}"
    return device


def resolve_device_for_model(
    model_type: str,
    preferred_device: str | None,
    legacy_device: str | None,
    gpu_index: int | None,
) -> str:
    if model_type in LEGACY_MODEL_TYPES:
        base_device = legacy_device or "cuda"
    elif preferred_device:
        base_device = preferred_device
    else:
        base_device = "cuda" if torch.cuda.is_available() else "cpu"
    return apply_gpu_index(base_device, gpu_index)


def resolve_enable_cueq(device: str, override: bool | None) -> bool:
    if override is not None:
        return override
    return DEFAULT_ENABLE_CUEQ and device.startswith("cuda")


def rounded_tuple(values, decimals: int = 8) -> tuple[float, ...]:
    return tuple(np.asarray(values, dtype=float).round(decimals).ravel())


def atom_signature(atoms, decimals: int = 8) -> tuple:
    return (
        atoms.info.get("nonpolar_mpid"),
        atoms.info.get("polar_mpid"),
        tuple(int(x) for x in atoms.numbers),
        rounded_tuple(atoms.cell.array, decimals),
        rounded_tuple(atoms.positions, decimals),
    )


def load_atoms(path: Path):
    return read(path, ":")


def result_value(results: dict, *keys: str):
    for key in keys:
        if key in results:
            return results[key]
    raise KeyError(f"None of {keys} found in calculator results: {sorted(results)}")


def build_mace_calculator(
    model_path: Path,
    calculator_cls,
    *,
    model_type: str,
    device: str,
    default_dtype: str,
    enable_cueq: bool,
    head: str,
    electric_field,
):
    kwargs = {
        "model_paths": str(model_path),
        "device": device,
        "default_dtype": default_dtype,
        "model_type": model_type,
        "enable_cueq": enable_cueq,
        "head": head,
    }
    if model_type in LEGACY_MODEL_TYPES:
        kwargs["electric_field"] = torch.as_tensor(
            (0.0, 0.0, 0.0) if electric_field is None else electric_field,
            dtype=torch.get_default_dtype(),
            device=device,
        )
    elif electric_field is not None:
        kwargs["electric_field"] = electric_field
    return calculator_cls(**kwargs)


def evaluate_atoms_with_model(
    atoms_list,
    model_path: Path,
    calculator_cls,
    root: Path,
    *,
    output_path: Path | None,
    force: bool,
    model_type: str,
    device: str,
    default_dtype: str,
    enable_cueq: bool,
    head: str,
    electric_field,
):
    if output_path is not None and output_path.exists() and not force:
        if output_path.stat().st_size == 0:
            print(f"Found empty cache file at {output_path.relative_to(root)}; recomputing.")
            output_path.unlink()
        else:
            try:
                print(f"Loading cached predictions from {output_path.relative_to(root)}")
                cached = read(output_path, ":")
                if len(cached) == len(atoms_list):
                    return cached
                print(
                    f"Cache length mismatch for {output_path.name}: "
                    f"{len(cached)} cached vs {len(atoms_list)} expected. Recomputing."
                )
                output_path.unlink()
            except Exception as exc:
                print(f"Could not read cache at {output_path.relative_to(root)} ({exc}); recomputing.")
                output_path.unlink(missing_ok=True)

    calc = build_mace_calculator(
        model_path,
        calculator_cls,
        model_type=model_type,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
        head=head,
        electric_field=electric_field,
    )

    predicted = []
    iterator = tqdm(atoms_list, total=len(atoms_list), desc=f"Evaluating {model_path.stem}", unit="cfg")
    for atoms in iterator:
        atoms_eval = atoms.copy()
        atoms_eval.calc = calc
        try:
            _ = atoms_eval.get_potential_energy()
        except RuntimeError as exc:
            message = str(exc)
            if model_type in LEGACY_MODEL_TYPES and device.startswith("cuda") and "libnvrtc-builtins" in message:
                raise RuntimeError(
                    "Legacy ScaleShiftFieldMACE hit a CUDA/NVRTC runtime error. "
                    "Check that LD_LIBRARY_PATH includes the conda env NVRTC path."
                ) from exc
            raise

        atoms_out = atoms.copy()
        results = atoms_eval.calc.results
        atoms_out.info["MACE_energy"] = float(result_value(results, "energy"))
        atoms_out.info["MACE_polarization"] = np.asarray(
            result_value(results, "polarization", "polarisation"), dtype=float
        ).reshape(3).copy()
        predicted.append(atoms_out)

    if output_path is not None:
        tmp_output = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
        if tmp_output.exists():
            tmp_output.unlink()
        write(tmp_output, predicted, format="extxyz", write_results=False)
        tmp_output.replace(output_path)
        print(f"Saved predictions to {output_path.relative_to(root)}")

    return predicted


def subset_by_reference(reference_atoms, predicted_atoms):
    lookup = defaultdict(list)
    for atoms in predicted_atoms:
        lookup[atom_signature(atoms)].append(atoms)

    subset = []
    for atoms in reference_atoms:
        key = atom_signature(atoms)
        matches = lookup.get(key)
        if not matches:
            raise KeyError(f"Could not match reference structure with key {key[:2]}")
        subset.append(matches.pop(0))
    return subset


def to_tensor(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float64)


def polarization_tensor(atoms_list, key: str) -> torch.Tensor:
    return to_tensor([np.asarray(atoms.info[key], dtype=float) for atoms in atoms_list])


def cell_tensor(atoms_list) -> torch.Tensor:
    return to_tensor([atoms.cell.array for atoms in atoms_list])


def flatten_components(ref_pol, pred_pol, which: str = "all") -> tuple[np.ndarray, np.ndarray]:
    ref_pol = np.asarray(ref_pol, dtype=float)
    pred_pol = np.asarray(pred_pol, dtype=float)
    if ref_pol.shape != pred_pol.shape:
        raise ValueError(f"Shape mismatch: {ref_pol.shape} vs {pred_pol.shape}")

    if ref_pol.ndim == 1:
        ref_view = ref_pol
        pred_view = pred_pol
    elif ref_pol.ndim == 2 and ref_pol.shape[1] != 3:
        ref_view = ref_pol.ravel()
        pred_view = pred_pol.ravel()
    elif ref_pol.ndim == 3:
        ref_view = ref_pol[:, -1, :] if which == "end" else ref_pol.reshape(-1, 3)
        pred_view = pred_pol[:, -1, :] if which == "end" else pred_pol.reshape(-1, 3)
    elif ref_pol.ndim == 2 and ref_pol.shape[1] == 3:
        ref_view = ref_pol
        pred_view = pred_pol
    else:
        ref_view = ref_pol.ravel()
        pred_view = pred_pol.ravel()

    x = ref_view.ravel()
    y = pred_view.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def parity_metrics(ref_values, pred_values) -> dict[str, float]:
    x, y = flatten_components(ref_values, pred_values)
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        slope = np.nan
        intercept = np.nan
        r2 = np.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def robust_symmetric_limits(ref_values, pred_values, pct_limits=(1, 99)) -> tuple[float, float]:
    x, y = flatten_components(ref_values, pred_values)
    combined = np.concatenate([x, y])
    q_low, q_high = np.percentile(combined, pct_limits)
    limit = max(abs(q_low), abs(q_high))
    limit = 1.05 * limit if limit > 0 else 1.0
    return (-limit, limit)


def plot_polarization_parity_splits(
    splits,
    *,
    units="μC/cm²",
    which="all",
    figsize=(4.9, 4.9),
    pct_limits=(1, 99),
    colors=None,
    markers=None,
    order=("train", "valid", "test"),
    s_train=10,
    s_valid=10,
    s_test=20,
    alpha_train=1.0,
    alpha_valtest=1.0,
    edgecolor="black",
    edgewidth=0.35,
    lw_guide=1.2,
    lw_fit=1.5,
    grid=True,
    max_points_per_split=200_000,
    annotate=True,
):
    if colors is None:
        colors = {"train": "#9E9E9E", "valid": "#4C78A8", "test": "#F58518"}
    if markers is None:
        markers = {"train": "o", "valid": "o", "test": "s"}

    def to_xy(ref_pol, pred_pol):
        return flatten_components(ref_pol, pred_pol, which=which)

    xy = {split: to_xy(*pair) for split, pair in splits.items()}
    all_vals = np.concatenate([np.r_[xy[split][0], xy[split][1]] for split in xy])
    q_low, q_high = np.percentile(all_vals[np.isfinite(all_vals)], pct_limits)
    limit = max(abs(q_low), abs(q_high))
    limit = 1.06 * limit if limit > 0 else 1.0
    lim = (-limit, limit)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(lim, lim, ls="--", color="0.45", lw=lw_guide, zorder=0)

    handles = []
    for split in order:
        if split not in xy:
            continue
        x, y = xy[split]
        if x.size > max_points_per_split:
            idx = np.random.default_rng(0).choice(x.size, size=max_points_per_split, replace=False)
            x = x[idx]
            y = y[idx]

        color = colors.get(split, "#333333")
        marker = markers.get(split, "o")
        size = s_train if split == "train" else (s_valid if split == "valid" else s_test)
        alpha = alpha_train if split == "train" else alpha_valtest

        ax.scatter(
            x,
            y,
            s=size,
            c=color,
            alpha=alpha,
            marker=marker,
            edgecolors=edgecolor,
            linewidths=edgewidth,
            zorder=1.5,
        )

        metrics = parity_metrics(x, y)
        fit_x = np.linspace(*lim, 200)
        ax.plot(fit_x, metrics["slope"] * fit_x + metrics["intercept"], color=color, lw=lw_fit, alpha=0.95)

        label = (
            f"{split}: $R^2={metrics['r2']:.3f}$, RMSE={metrics['rmse']:.2g}, MAE={metrics['mae']:.2g}"
            if annotate
            else split
        )
        handles.append(plt.Line2D([], [], color=color, lw=lw_fit, label=label))

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, which="major", ls="--", lw=0.6, color="0.9")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=2)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel(f"REF polarization ({units})" if units else "REF polarization")
    ax.set_ylabel(f"MACE polarization ({units})" if units else "MACE polarization")
    set_axis_fontsizes(ax)
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.8, borderaxespad=0.3)
    return fig, ax


def plot_polarization_parity_components(
    ref_pol,
    pred_pol,
    *,
    units="μC/cm²",
    which="all",
    gridsize=55,
    cmap="viridis",
    pct_limits=(1, 99),
    annotate="short",
    figsize=(10.5, 3.8),
    lw_guide=1.1,
    lw_fit=1.3,
):
    ref_pol = np.asarray(ref_pol, dtype=float)
    pred_pol = np.asarray(pred_pol, dtype=float)
    if ref_pol.shape != pred_pol.shape:
        raise ValueError(f"Shape mismatch: {ref_pol.shape} vs {pred_pol.shape}")

    if ref_pol.ndim == 3:
        ref_view = ref_pol[:, -1, :] if which == "end" else ref_pol.reshape(-1, 3)
        pred_view = pred_pol[:, -1, :] if which == "end" else pred_pol.reshape(-1, 3)
    elif ref_pol.ndim == 2 and ref_pol.shape[1] == 3:
        ref_view = ref_pol
        pred_view = pred_pol
    else:
        raise ValueError("Expected arrays shaped (N,3) or (Nmat,Nsteps,3).")

    lim = robust_symmetric_limits(ref_view, pred_view, pct_limits=pct_limits)
    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)
    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.02, top=0.90, wspace=0.35)

    labels = ["x component", "y component", "z component"]
    mappable = None
    for idx, ax in enumerate(axs):
        x = ref_view[:, idx]
        y = pred_view[:, idx]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        hb = ax.hexbin(
            x,
            y,
            gridsize=gridsize,
            extent=(*lim, *lim),
            bins="log",
            cmap=cmap,
            mincnt=1,
            rasterized=True,
        )
        mappable = hb
        ax.plot(lim, lim, "--", color="0.35", lw=lw_guide, zorder=1)
        if x.size >= 2:
            metrics = parity_metrics(x, y)
            fit_x = np.linspace(*lim, 2)
            ax.plot(fit_x, metrics["slope"] * fit_x + metrics["intercept"], color="tab:orange", lw=lw_fit, alpha=0.95)
            if annotate and annotate.lower() != "none":
                text = f"$R^2={metrics['r2']:.3f}$\nRMSE={metrics['rmse']:.3g}\nMAE={metrics['mae']:.3g}"
                ax.text(
                    0.02,
                    0.98,
                    text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=PLOT_FONTSIZE,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
                )

        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[idx], pad=3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", length=2)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        set_axis_fontsizes(ax)

    fig.supxlabel(f"REF polarization ({units})" if units else "REF polarization", fontsize=PLOT_FONTSIZE)
    fig.supylabel(f"MACE polarization ({units})" if units else "MACE polarization", fontsize=PLOT_FONTSIZE)
    cax = fig.add_axes([0.90, 0.22, 0.015, 0.60])
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cb.set_label("log density", fontsize=PLOT_FONTSIZE)
    cb.ax.tick_params(labelsize=PLOT_FONTSIZE)
    cb.formatter = LogFormatter(10, labelOnlyBase=False)
    cb.update_ticks()
    return fig, axs


def plot_spontaneous_polarization_parity(
    ref_ps,
    pred_ps,
    *,
    units="μC/cm²",
    absolute=True,
    gridsize=55,
    cmap="viridis",
    pct_limits=(1, 99),
    annotate="short",
    figsize=(4.6, 4.6),
    lw_guide=1.1,
    lw_fit=1.3,
):
    x = np.ravel(np.asarray(ref_ps, dtype=float))
    y = np.ravel(np.asarray(pred_ps, dtype=float))
    if absolute:
        x = np.abs(x)
        y = np.abs(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No finite values in inputs.")

    vals = np.r_[x, y]
    q_low, q_high = np.percentile(vals, pct_limits)
    if absolute:
        low, high = 0.0, max(q_high, 0.0)
    else:
        limit = max(abs(q_low), abs(q_high))
        low, high = -limit, limit
    pad = 1.05 if high > 0 else 1.0
    lim = (low, pad * high if absolute else pad * high)

    metrics = parity_metrics(x, y)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=False)
    plt.subplots_adjust(left=0.18, right=0.86, bottom=0.16, top=0.92)
    hb = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        extent=(*lim, *lim),
        bins="log",
        cmap=cmap,
        mincnt=1,
        rasterized=True,
    )
    ax.plot(lim, lim, "--", color="0.35", lw=lw_guide, zorder=1)
    fit_x = np.linspace(*lim, 2)
    ax.plot(fit_x, metrics["slope"] * fit_x + metrics["intercept"], color="tab:orange", lw=lw_fit, alpha=0.95)
    if annotate and annotate.lower() != "none":
        text = f"$R^2={metrics['r2']:.3f}$\nRMSE={metrics['rmse']:.3g}\nMAE={metrics['mae']:.3g}"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PLOT_FONTSIZE,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=2)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel(f"REF spontaneous polarization ({units})")
    ax.set_ylabel(f"MACE spontaneous polarization ({units})")
    set_axis_fontsizes(ax)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log density", fontsize=PLOT_FONTSIZE)
    cb.ax.tick_params(labelsize=PLOT_FONTSIZE)
    cb.formatter = LogFormatter(10, labelOnlyBase=False)
    cb.update_ticks()
    return fig, ax


def consecutive_path_groups(atoms_list):
    keys = [(atoms.info["nonpolar_mpid"], atoms.info["polar_mpid"]) for atoms in atoms_list]
    ranges = []
    group_keys = []
    start = 0
    while start < len(keys):
        key = keys[start]
        stop = start + 1
        while stop < len(keys) and keys[stop] == key:
            stop += 1
        ranges.append((start, stop))
        group_keys.append(key)
        start = stop
    return ranges, group_keys


def build_qpol(cells: torch.Tensor) -> torch.Tensor:
    polar_cells = cells[:, -1, :, :]
    volumes = torch.linalg.det(polar_cells).abs().clamp_min(1e-12).view(-1, 1, 1)
    return polar_cells / volumes


def unwrap_polarization_paths(polarization, qpol, fold_polarization):
    polarization = to_tensor(polarization)
    qpol = to_tensor(qpol)
    n_paths, n_steps, _ = polarization.shape
    if n_steps == 1:
        return torch.zeros_like(polarization), torch.zeros_like(polarization)

    q_steps = qpol[:, None, :, :].expand(n_paths, n_steps - 1, 3, 3).reshape(-1, 3, 3)
    delta, _ = fold_polarization(
        polarization[:, 1:, :].reshape(-1, 3),
        polarization[:, :-1, :].reshape(-1, 3),
        q_steps,
    )
    delta = delta.reshape(n_paths, n_steps - 1, 3)

    unwrapped = torch.zeros_like(polarization)
    unwrapped[:, 1:, :] = torch.cumsum(delta, dim=1)

    q_full = qpol[:, None, :, :].expand(n_paths, n_steps, 3, 3)
    fractional = torch.linalg.solve(q_full.transpose(-2, -1), unwrapped.unsqueeze(-1)).squeeze(-1)
    return unwrapped, fractional


def plot_path_components(
    ref_paths,
    pred_paths,
    *,
    index: int,
    formula: str | None = None,
    title_suffix: str | None = None,
    units="μC/cm²",
    figsize=(6.0, 4.0),
    show_magnitude=True,
    tick_decimals=1,
):
    ref_path = np.asarray(ref_paths[index], dtype=float)
    pred_path = np.asarray(pred_paths[index], dtype=float)
    n_steps = ref_path.shape[0]
    pos = np.arange(n_steps)
    tick_labels = [f"{value:.{tick_decimals}f}" for value in np.linspace(0.0, 1.0, n_steps)]

    colors = {"x": "#4C78A8", "y": "#F58518", "z": "#54A24B"}
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=False)
    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.88)

    for comp_idx, comp in enumerate(("x", "y", "z")):
        ax.plot(pos, ref_path[:, comp_idx], marker="o", ms=4.2, lw=1.9, label=f"REF {comp}", color=colors[comp])
        ax.plot(
            pos,
            pred_path[:, comp_idx],
            marker="s",
            ms=4.0,
            lw=1.7,
            ls="--",
            mfc="white",
            mec=colors[comp],
            mew=1.0,
            label=f"MACE {comp}",
            color=colors[comp],
        )

    if show_magnitude:
        ref_mag = np.linalg.norm(ref_path, axis=1)
        pred_mag = np.linalg.norm(pred_path, axis=1)
        ax.plot(pos, ref_mag, color="0.25", lw=1.3, ls="-.", label="REF |P|")
        ax.plot(pos, pred_mag, color="0.25", lw=1.3, ls=":", label="MACE |P|")

    for step in pos:
        ax.axvline(step, color="0.85", ls="--", lw=0.7, zorder=0)
    ax.axhline(0.0, color="0.35", lw=1.0)

    title = formula if formula else f"path {index}"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    ax.set_title(title)
    ax.set_xlim(-0.35, n_steps - 1 + 0.35)
    ax.set_xlabel("distortion parameter")
    ax.set_ylabel(f"polarization ({units})")
    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels)
    ax.legend(frameon=False, ncol=2, loc="best")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=2)
    set_axis_fontsizes(ax)
    return fig, ax


def plot_fractional_path_distribution(
    ref_fractional,
    pred_fractional,
    *,
    figsize=(4.6, 5.5),
    limits=(-0.6, 0.6),
    colors=("#4C78A8", "#F58518"),
    labels=("REF", "MACE"),
    violin_alpha=0.32,
    violin_edge="white",
    fan_alpha=(0.10, 0.22),
    center_gap=0.012,
):
    ref_fractional = np.asarray(ref_fractional, dtype=float)
    pred_fractional = np.asarray(pred_fractional, dtype=float)
    if ref_fractional.shape != pred_fractional.shape or ref_fractional.ndim != 3:
        raise ValueError(f"Expected (M,T,3) arrays; got {ref_fractional.shape=} and {pred_fractional.shape=}")

    n_paths, n_steps, _ = ref_fractional.shape
    positions = np.arange(n_steps, dtype=float)
    tick_labels = [f"{value:.1f}" for value in np.linspace(0.0, 1.0, n_steps)]
    pooled_ref = np.transpose(ref_fractional, (0, 2, 1)).reshape(n_paths * 3, n_steps)
    pooled_pred = np.transpose(pred_fractional, (0, 2, 1)).reshape(n_paths * 3, n_steps)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=False)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.88)
    for step in positions:
        ax.axvline(step, color="0.88", ls="--", lw=0.7, zorder=0)

    def draw_fan(values, color):
        qs = np.quantile(values, [0.05, 0.25, 0.75, 0.95], axis=0)
        ax.fill_between(positions, qs[0], qs[3], color=color, alpha=fan_alpha[0], zorder=0.4)
        ax.fill_between(positions, qs[1], qs[2], color=color, alpha=fan_alpha[1], zorder=0.5)

    draw_fan(pooled_ref, colors[0])
    draw_fan(pooled_pred, colors[1])

    width = 0.66
    parts_left = ax.violinplot(
        dataset=[pooled_ref[:, idx] for idx in range(n_steps)],
        positions=positions,
        widths=width,
        showmeans=False,
        showextrema=False,
        bw_method="scott",
        points=200,
    )
    parts_right = ax.violinplot(
        dataset=[pooled_pred[:, idx] for idx in range(n_steps)],
        positions=positions,
        widths=width,
        showmeans=False,
        showextrema=False,
        bw_method="scott",
        points=200,
    )

    def clip_half(body, position, side, offset):
        vertices = body.get_paths()[0].vertices
        x_coords = vertices[:, 0]
        x_coords = np.minimum(x_coords, position) if side == "left" else np.maximum(x_coords, position)
        vertices[:, 0] = x_coords + offset

    for idx, body in enumerate(parts_left["bodies"]):
        body.set_facecolor(colors[0])
        body.set_alpha(violin_alpha)
        body.set_edgecolor(violin_edge)
        body.set_linewidth(0.6)
        body.set_zorder(1.2)
        clip_half(body, positions[idx], "left", -0.5 * center_gap)

    for idx, body in enumerate(parts_right["bodies"]):
        body.set_facecolor(colors[1])
        body.set_alpha(violin_alpha)
        body.set_edgecolor(violin_edge)
        body.set_linewidth(0.6)
        body.set_zorder(1.2)
        clip_half(body, positions[idx], "right", 0.5 * center_gap)

    ax.axhline(0.0, color="0.3", lw=1.0, zorder=2)
    ax.axhline(+0.5, color="0.6", ls=":", lw=1.0, zorder=2)
    ax.axhline(-0.5, color="0.6", ls=":", lw=1.0, zorder=2)
    ax.set_xlim(-0.4, n_steps - 1 + 0.4)
    ax.set_ylim(*limits)
    ax.set_title("All polarization branches")
    ax.set_xlabel("Distortion parameter")
    ax.set_ylabel("Fraction of polarization quantum")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(
        handles=[
            Patch(facecolor=colors[0], alpha=fan_alpha[1], label=labels[0]),
            Patch(facecolor=colors[1], alpha=fan_alpha[1], label=labels[1]),
        ],
        frameon=False,
        loc="upper left",
    )
    set_axis_fontsizes(ax)
    return fig, ax


def plot_polar_branch_ladder_compare(
    P_ref,
    P_mace,
    Qpol,
    index,
    *,
    component="z",
    units=r"$\mu\mathrm{C}/\mathrm{cm}^2$",
    n_branches=None,
    extra=3,
    figsize=(6.2, 8.0),
    formula=None,
    colours=None,
):
    P_ref = P_ref if isinstance(P_ref, torch.Tensor) else torch.tensor(P_ref)
    P_mace = P_mace if isinstance(P_mace, torch.Tensor) else torch.tensor(P_mace)
    Qpol = Qpol if isinstance(Qpol, torch.Tensor) else torch.tensor(Qpol)

    def _fold(dP, Q):
        frac = torch.linalg.solve(Q, dP.unsqueeze(-1)).squeeze(-1)
        frac = frac - torch.floor(frac + 0.5)
        return (Q @ frac.unsqueeze(-1)).squeeze(-1)

    def _unwrap(P, Q):
        T = P.shape[0]
        out = torch.zeros_like(P)
        if T > 1:
            steps = _fold(P[1:] - P[:-1], Q)
            out[1:] = torch.cumsum(steps, dim=0)
        return out

    Q = Qpol if Qpol.ndim == 2 else Qpol[index]
    Prefu = _unwrap(P_ref[index], Q).detach().cpu().numpy()
    Pmacu = _unwrap(P_mace[index], Q).detach().cpu().numpy()

    comp = {"x": 0, "y": 1, "z": 2}.get(component, component)
    y_ref, y_mace = Prefu[:, comp], Pmacu[:, comp]
    q = float(abs(Q[comp, comp].item()))
    if q < 1e-12:
        raise ValueError("Component polarization quantum is too small.")

    if n_branches is None:
        low = min(y_ref.min(), y_mace.min()) / q
        high = max(y_ref.max(), y_mace.max()) / q
        bmin, bmax = int(np.floor(low)) - extra, int(np.ceil(high)) + extra
        branches = list(range(bmin, bmax + 1))
    else:
        low = min(y_ref.min(), y_mace.min()) / q
        high = max(y_ref.max(), y_mace.max()) / q
        center = int(np.round(0.5 * (np.floor(low) + np.ceil(high))))
        half = (n_branches - 1) // 2
        branches = list(range(center - half, center + half + 1))

    T = y_ref.size
    pos = np.arange(T)
    xticklabels = [f"{t:.1f}" for t in np.linspace(0.0, 1.0, T)]
    if colours is None:
        base = ["#5F0F40", "#9A031E", "#FB8B24", "#E36414", "#0F4C5C", "#2A9D8F", "#3A86FF", "#8338EC"]
        colours = (base * ((len(branches) // len(base)) + 1))[: len(branches)]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.90)
    for branch in branches:
        ax.axhline(branch * q, ls="--", lw=0.9, color="0.80", zorder=0)

    first = True
    for color, branch in zip(colours, branches):
        yR = y_ref + branch * q
        yM = y_mace + branch * q
        ax.plot(pos, yR, color=color, lw=2.0, marker="o", ms=4.0, label="Reference" if first else None)
        ax.plot(
            pos,
            yM,
            color=color,
            lw=1.9,
            ls="--",
            dashes=(4, 2),
            marker="s",
            ms=3.8,
            mfc="white",
            mec=color,
            mew=1.0,
            label="MACE" if first else None,
        )
        first = False

    for pk in pos:
        ax.axvline(pk, color="0.90", ls=":", lw=0.7, zorder=0)

    ymin = (min(branches) - 1) * q
    ymax = (max(branches) + 1) * q
    pad = 0.06 * max(1e-12, ymax - ymin)
    ax.set_xlim(-0.35, T - 1 + 0.35)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title(f"{formula} branches" if formula else "branches")
    ax.set_xlabel("distortion parameter")
    ax.set_ylabel(rf"Polarisation along $\hat{{{component}}}$ ({units})")
    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels)
    ax.legend(frameon=False, loc="best", ncol=2)
    set_axis_fontsizes(ax)
    return fig, ax


def save_figure(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_analysis_tables(
    figure_dir: Path,
    model_stem: str,
    summary: pd.DataFrame,
    materials: pd.DataFrame,
    sp_summary: pd.DataFrame,
    path_sp_df: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> None:
    summary.to_csv(figure_dir / f"{model_stem}_polarization_summary.csv", index=False)
    materials.to_csv(figure_dir / f"{model_stem}_materials.csv", index=False)
    sp_summary.to_csv(figure_dir / f"{model_stem}_spontaneous_polarization_summary.csv", index=False)
    path_sp_df.to_csv(figure_dir / f"{model_stem}_spontaneous_polarization_by_material.csv", index=False)
    np.savez_compressed(figure_dir / f"{model_stem}_analysis_data.npz", **arrays)


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    root = resolve_root(args.root)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    figure_dir = output_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = figure_dir.parent if figure_dir.name == "polarization_branches" else figure_dir
    dataset_paths = {
        "all": root / "MP-Ferroelectrics.xyz",
        "train": root / "MP-Ferroelectrics-train.xyz",
        "valid": root / "MP-Ferroelectrics-valid.xyz",
        "test": root / "MP-Ferroelectrics-test.xyz",
    }
    return RuntimeConfig(
        root=root,
        analysis_dir=analysis_dir,
        figure_dir=figure_dir,
        dataset_paths=dataset_paths,
        material_indices=args.material_indices,
        force_rerun=args.force_rerun,
        default_dtype=args.default_dtype,
        electric_field=None,
        preferred_device=args.device,
        legacy_device=args.legacy_device,
        gpu_index=args.gpu,
        enable_cueq=DEFAULT_ENABLE_CUEQ,
        cueq_override=args.cueq_override,
        head_override=args.head,
        model_type_override=args.model_type,
        code_root_override=args.code_root,
        plot_dpi=args.plot_dpi,
    )


def run_analysis_for_model(model_path: Path, runtime: RuntimeConfig) -> dict[str, object]:
    model_family = detect_model_family_from_file(model_path)
    code_root = resolve_code_root(model_family, runtime.code_root_override)
    calculator_cls, fold_polarization = load_mace_api(code_root)
    model_info = inspect_model_file(model_path)
    model_type = runtime.model_type_override or model_info["model_type"]
    head = runtime.head_override or ("Default" if model_type in LEGACY_MODEL_TYPES else "pt_head")
    device = resolve_device_for_model(
        model_type,
        runtime.preferred_device,
        runtime.legacy_device,
        runtime.gpu_index,
    )
    enable_cueq = resolve_enable_cueq(device, runtime.cueq_override)
    predicted_all_path = runtime.analysis_dir / f"{model_path.stem}__MP-Ferroelectric.predicted.extxyz"

    model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="minutes")
    print(
        "\n".join(
            [
                f"Model: {model_path.name}",
                f"  timestamp: {model_mtime}",
                f"  family: {model_family}",
                f"  code root: {code_root}",
                f"  class: {model_info['class']}",
                f"  calculator model type: {model_type}",
                f"  head: {head}",
                f"  device: {device}",
                f"  CuEq enabled: {enable_cueq}",
                f"  prediction cache: {predicted_all_path}",
            ]
        )
    )

    reference_atoms = {split: load_atoms(path) for split, path in runtime.dataset_paths.items()}
    predicted_atoms = {
        "all": evaluate_atoms_with_model(
            reference_atoms["all"],
            model_path,
            calculator_cls,
            runtime.root,
            output_path=predicted_all_path,
            force=runtime.force_rerun,
            model_type=model_type,
            device=device,
            default_dtype=runtime.default_dtype,
            enable_cueq=enable_cueq,
            head=head,
            electric_field=runtime.electric_field,
        )
    }
    for split in ("train", "valid", "test"):
        predicted_atoms[split] = subset_by_reference(reference_atoms[split], predicted_atoms["all"])

    ref_pol_raw = {split: polarization_tensor(reference_atoms[split], "REF_polarization") for split in runtime.dataset_paths}
    pred_pol_raw = {split: polarization_tensor(predicted_atoms[split], "MACE_polarization") for split in runtime.dataset_paths}
    cells = {split: cell_tensor(reference_atoms[split]) for split in runtime.dataset_paths}

    pred_pol_folded_raw = {}
    for split in runtime.dataset_paths:
        delta, _ = fold_polarization(pred_pol_raw[split], ref_pol_raw[split], cells[split])
        pred_pol_folded_raw[split] = delta + ref_pol_raw[split]

    ref_pol = {split: ref_pol_raw[split] * POL_UC_CM2 for split in runtime.dataset_paths}
    pred_pol = {split: pred_pol_raw[split] * POL_UC_CM2 for split in runtime.dataset_paths}
    pred_pol_folded = {split: pred_pol_folded_raw[split] * POL_UC_CM2 for split in runtime.dataset_paths}

    summary_rows = []
    for split in ("train", "valid", "test", "all"):
        raw_metrics = parity_metrics(ref_pol[split], pred_pol[split])
        folded_metrics = parity_metrics(ref_pol[split], pred_pol_folded[split])
        summary_rows.append(
            {
                "split": split,
                "raw_rmse": raw_metrics["rmse"],
                "raw_mae": raw_metrics["mae"],
                "raw_r2": raw_metrics["r2"],
                "folded_rmse": folded_metrics["rmse"],
                "folded_mae": folded_metrics["mae"],
                "folded_r2": folded_metrics["r2"],
            }
        )
    summary = pd.DataFrame(summary_rows)

    raw_splits = {split: (ref_pol[split], pred_pol[split]) for split in ("train", "valid", "test")}
    fig, _ = plot_polarization_parity_splits(raw_splits, which="all")
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_parity_raw_splits.png", runtime.plot_dpi)

    fig, _ = plot_polarization_parity_components(ref_pol["all"], pred_pol["all"], which="all")
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_parity_raw_components.png", runtime.plot_dpi)

    folded_splits = {split: (ref_pol[split], pred_pol_folded[split]) for split in ("train", "valid", "test")}
    fig, _ = plot_polarization_parity_splits(folded_splits, which="all")
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_parity_folded_splits.png", runtime.plot_dpi)

    fig, _ = plot_polarization_parity_components(ref_pol["all"], pred_pol_folded["all"], which="all")
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_parity_folded_components.png", runtime.plot_dpi)

    path_ranges, path_keys = consecutive_path_groups(reference_atoms["all"])
    path_sizes = [stop - start for start, stop in path_ranges]
    if len(set(path_sizes)) != 1:
        raise ValueError(f"Inconsistent path lengths: {sorted(set(path_sizes))}")

    n_paths = len(path_ranges)
    n_steps = path_sizes[0]
    ref_paths_raw = ref_pol_raw["all"].reshape(n_paths, n_steps, 3)
    pred_paths_raw = pred_pol_raw["all"].reshape(n_paths, n_steps, 3)
    ref_paths = ref_pol["all"].reshape(n_paths, n_steps, 3)
    pred_paths_folded = pred_pol_folded["all"].reshape(n_paths, n_steps, 3)
    cells_all = cells["all"].reshape(n_paths, n_steps, 3, 3)
    qpol = build_qpol(cells_all)

    split_signatures = {
        split: {atom_signature(atoms) for atoms in reference_atoms[split]}
        for split in ("train", "valid", "test")
    }

    def infer_frame_split(atoms):
        signature = atom_signature(atoms)
        for split in ("train", "valid", "test"):
            if signature in split_signatures[split]:
                return split
        return None

    path_nonpolar_splits = []
    path_polar_splits = []
    path_splits = []
    for start, stop in path_ranges:
        nonpolar_split = infer_frame_split(reference_atoms["all"][start])
        polar_split = infer_frame_split(reference_atoms["all"][stop - 1])
        assigned_split = polar_split or nonpolar_split
        if assigned_split is None:
            raise KeyError(f"Could not assign path split for path starting at frame {start}")
        path_nonpolar_splits.append(nonpolar_split)
        path_polar_splits.append(polar_split)
        path_splits.append(assigned_split)

    materials = pd.DataFrame(
        [
            {
                "index": idx,
                "nonpolar_mpid": key[0],
                "polar_mpid": key[1],
                "formula": reference_atoms["all"][start].get_chemical_formula(mode="metal"),
                "split": split,
                "nonpolar_split": nonpolar_split,
                "polar_split": polar_split,
            }
            for idx, ((start, _), key, split, nonpolar_split, polar_split) in enumerate(
                zip(path_ranges, path_keys, path_splits, path_nonpolar_splits, path_polar_splits)
            )
        ]
    )

    for material_index in runtime.material_indices:
        if material_index < 0 or material_index >= len(materials):
            raise IndexError(f"material index {material_index} is out of range for {len(materials)} paths")
        material = materials.iloc[material_index]
        fig, _ = plot_polar_branch_ladder_compare(
            P_ref=ref_paths,
            P_mace=pred_paths_folded,
            Qpol=qpol * POL_UC_CM2,
            index=material_index,
            component="z",
            units=r"$\mu\mathrm{C}/\mathrm{cm}^2$",
            figsize=(5.0, 5.5),
            n_branches=8,
            formula=material["formula"],
        )
        save_figure(
            fig,
            runtime.figure_dir / f"{model_path.stem}_path_branches_{material_index:03d}.png",
            runtime.plot_dpi,
        )

        fig, _ = plot_path_components(
            ref_paths,
            pred_paths_folded,
            index=material_index,
            formula=material["formula"],
            title_suffix=f"{material['nonpolar_mpid']} -> {material['polar_mpid']}",
        )
        save_figure(
            fig,
            runtime.figure_dir / f"{model_path.stem}_path_components_{material_index:03d}.png",
            runtime.plot_dpi,
        )

    pred_delta_to_ref, _ = fold_polarization(
        pred_paths_raw.reshape(-1, 3),
        ref_paths_raw.reshape(-1, 3),
        cells_all.reshape(-1, 3, 3),
    )
    pred_paths_ref_aligned_raw = (pred_delta_to_ref + ref_paths_raw.reshape(-1, 3)).reshape(n_paths, n_steps, 3)
    ref_paths_unwrapped, ref_paths_fractional = unwrap_polarization_paths(ref_paths_raw, qpol, fold_polarization)
    pred_paths_ref_aligned_unwrapped, pred_paths_ref_aligned_fractional = unwrap_polarization_paths(
        pred_paths_ref_aligned_raw, qpol, fold_polarization
    )

    ref_sp_raw = (ref_paths_raw[:, -1, :] - ref_paths_raw[:, 0, :]) * POL_UC_CM2
    pred_sp_raw = (pred_paths_ref_aligned_raw[:, -1, :] - pred_paths_ref_aligned_raw[:, 0, :]) * POL_UC_CM2
    ref_sp_folded = (ref_paths_unwrapped[:, -1, :] - ref_paths_unwrapped[:, 0, :]) * POL_UC_CM2
    pred_sp_folded = (
        pred_paths_ref_aligned_unwrapped[:, -1, :] - pred_paths_ref_aligned_unwrapped[:, 0, :]
    ) * POL_UC_CM2

    split_indices = {
        split: np.flatnonzero(materials["split"].to_numpy() == split)
        for split in ("train", "valid", "test")
    }
    sp_summary = pd.DataFrame(
        [
            {
                "metric": "mean |P_s|",
                "reference": torch.linalg.norm(ref_sp_folded, dim=1).mean().item(),
                "mace": torch.linalg.norm(pred_sp_folded, dim=1).mean().item(),
            },
            {
                "metric": "median |P_s|",
                "reference": torch.linalg.norm(ref_sp_folded, dim=1).median().item(),
                "mace": torch.linalg.norm(pred_sp_folded, dim=1).median().item(),
            },
        ]
    )

    path_sp_df = materials.copy()
    for axis, idx in zip(("x", "y", "z"), range(3)):
        path_sp_df[f"ref_sp_raw_{axis}"] = ref_sp_raw[:, idx].detach().cpu().numpy()
        path_sp_df[f"pred_sp_raw_{axis}"] = pred_sp_raw[:, idx].detach().cpu().numpy()
        path_sp_df[f"ref_sp_folded_{axis}"] = ref_sp_folded[:, idx].detach().cpu().numpy()
        path_sp_df[f"pred_sp_folded_{axis}"] = pred_sp_folded[:, idx].detach().cpu().numpy()
    path_sp_df["ref_sp_folded_norm"] = torch.linalg.norm(ref_sp_folded, dim=1).detach().cpu().numpy()
    path_sp_df["pred_sp_folded_norm"] = torch.linalg.norm(pred_sp_folded, dim=1).detach().cpu().numpy()

    sp_raw_splits = {split: (ref_sp_raw[idx], pred_sp_raw[idx]) for split, idx in split_indices.items() if len(idx) > 0}
    sp_folded_splits = {
        split: (ref_sp_folded[idx], pred_sp_folded[idx]) for split, idx in split_indices.items() if len(idx) > 0
    }

    fig, _ = plot_polarization_parity_splits(sp_raw_splits, units="μC/cm²", which="all")
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_spontaneous_polarization_raw_splits.png", runtime.plot_dpi)

    fig, _ = plot_polarization_parity_splits(sp_folded_splits, units="μC/cm²", which="all")
    save_figure(
        fig,
        runtime.figure_dir / f"{model_path.stem}_spontaneous_polarization_folded_splits.png",
        runtime.plot_dpi,
    )

    fig, _ = plot_spontaneous_polarization_parity(
        ref_sp_folded,
        pred_sp_folded,
        absolute=False,
        gridsize=70,
        pct_limits=(1, 99.9),
    )
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_spontaneous_polarization.png", runtime.plot_dpi)

    fig, _ = plot_polarization_parity_components(
        ref_sp_folded,
        pred_sp_folded,
        units="μC/cm²",
        which="all",
        gridsize=70,
        pct_limits=(1, 99.9),
    )
    save_figure(
        fig,
        runtime.figure_dir / f"{model_path.stem}_spontaneous_polarization_components.png",
        runtime.plot_dpi,
    )

    fig, _ = plot_fractional_path_distribution(
        ref_paths_fractional,
        pred_paths_ref_aligned_fractional,
        limits=(-1.0, 1.0),
        center_gap=0.03,
        violin_alpha=0.9,
        violin_edge="grey",
        fan_alpha=(0.20, 0.32),
        figsize=(4.2, 5.5),
    )
    save_figure(fig, runtime.figure_dir / f"{model_path.stem}_fractional_branch_distribution.png", runtime.plot_dpi)

    save_analysis_tables(
        runtime.figure_dir,
        model_path.stem,
        summary,
        materials,
        sp_summary,
        path_sp_df,
        arrays={
            "ref_pol_all_uc_cm2": ref_pol["all"].detach().cpu().numpy(),
            "pred_pol_all_uc_cm2": pred_pol["all"].detach().cpu().numpy(),
            "pred_pol_folded_all_uc_cm2": pred_pol_folded["all"].detach().cpu().numpy(),
            "ref_paths_uc_cm2": ref_paths.detach().cpu().numpy(),
            "pred_paths_folded_uc_cm2": pred_paths_folded.detach().cpu().numpy(),
            "qpol_e_per_a2": qpol.detach().cpu().numpy(),
            "ref_sp_folded_uc_cm2": ref_sp_folded.detach().cpu().numpy(),
            "pred_sp_folded_uc_cm2": pred_sp_folded.detach().cpu().numpy(),
            "ref_paths_fractional": ref_paths_fractional.detach().cpu().numpy(),
            "pred_paths_fractional": pred_paths_ref_aligned_fractional.detach().cpu().numpy(),
        },
    )

    return {
        "model": model_path.name,
        "summary": summary,
        "sp_summary": sp_summary,
        "n_paths": len(materials),
        "n_structures": len(reference_atoms["all"]),
    }


def main() -> None:
    args = parse_args()
    runtime = build_runtime_config(args)
    model_paths = resolve_model_inputs(runtime.root, args.models)

    results = []
    for model_path in model_paths:
        results.append(run_analysis_for_model(model_path, runtime))

    combined_rows = []
    for result in results:
        summary = result["summary"].copy()
        summary.insert(0, "model", result["model"])
        combined_rows.append(summary)
    combined_summary = pd.concat(combined_rows, ignore_index=True)
    combined_path = runtime.figure_dir / "combined_polarization_summary.csv"
    combined_summary.to_csv(combined_path, index=False)

    print(f"\nSaved combined summary to {combined_path}")
    print(f"Saved plots and data to {runtime.figure_dir}")


if __name__ == "__main__":
    main()
