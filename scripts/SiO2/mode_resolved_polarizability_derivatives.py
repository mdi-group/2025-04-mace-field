#!/usr/bin/env python3
"""Mode-resolved polarizability-derivative analysis for alpha-quartz.

This script compares the direct SiO2 model against the cross-chemistry
foundation model in two complementary ways:

1. Native mode analysis:
   Each model uses its own relaxed zero-field structure and Hessian to define
   Gamma-point normal modes. Finite-difference polarizability derivatives
   ``d alpha / dQ_m`` are evaluated for every optical mode, and Raman-relevant
   invariants are reported.

2. Common-basis analysis:
   The direct model's relaxed structure and normal-mode basis are reused for
   both response models. This removes the foundation model's softened PES from
   the comparison and isolates how well the transferred polarizability response
   matches the direct quartz model on the same modal coordinates.

Outputs are written as compact CSV / JSON tables plus a few summary plots that
are easy to cite in the manuscript or reuse in notebooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.ticker import AutoMinorLocator
from mace.calculators import MACECalculator

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE.parent / "LAMMPs" / "MD" / "runs"
DEFAULT_OUTPUT_DIR = HERE / "plots" / "mode_resolved_polarizability"

DEFAULT_DIRECT_MODEL = HERE / "MACE-field-SiO2.model"
DEFAULT_FOUNDATION_MODEL = HERE.parent / "Foundation" / "MACEField-omat-dielectric.model"
DEFAULT_DIRECT_HEAD = "Default"
DEFAULT_FOUNDATION_ENERGY_HEAD = "pt_head"
DEFAULT_FOUNDATION_RESPONSE_HEAD = "mp-dielectric"

EV_J = 1.602176634e-19
ANG_M = 1.0e-10
AMU_KG = 1.66053906660e-27
LIGHT_CM_S = 2.99792458e10

DIRECT_COLOR = "#0055d4"
FOUNDATION_COLOR = "#d45500"
COMMON_COLOR = "#14805e"


@dataclass(frozen=True)
class ModelAnalysisConfig:
    label: str
    model_path: Path
    energy_head: str
    response_head: str
    structure_path: Path
    color: str


@dataclass
class CalculatorBundle:
    energy_calc: MACECalculator
    response_calc: MACECalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct and foundation mode-resolved polarizability derivatives for alpha-quartz."
    )
    parser.add_argument("--direct-model", type=Path, default=DEFAULT_DIRECT_MODEL)
    parser.add_argument("--direct-energy-head", default=DEFAULT_DIRECT_HEAD)
    parser.add_argument("--direct-response-head", default=DEFAULT_DIRECT_HEAD)
    parser.add_argument("--direct-structure", type=Path, default=None)
    parser.add_argument("--foundation-model", type=Path, default=DEFAULT_FOUNDATION_MODEL)
    parser.add_argument("--foundation-energy-head", default=DEFAULT_FOUNDATION_ENERGY_HEAD)
    parser.add_argument("--foundation-response-head", default=DEFAULT_FOUNDATION_RESPONSE_HEAD)
    parser.add_argument("--foundation-structure", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Torch device for MACECalculator, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=("float32", "float64"),
        help="Inference dtype passed to the calculators.",
    )
    parser.add_argument(
        "--delta-q",
        type=float,
        default=0.10,
        help="Central-difference displacement amplitude in sqrt(amu) * Angstrom.",
    )
    parser.add_argument(
        "--spectrum-sigma-cm",
        type=float,
        default=20.0,
        help="Gaussian broadening applied to discrete Raman spectra.",
    )
    parser.add_argument(
        "--max-frequency-cm",
        type=float,
        default=1400.0,
        help="Upper x-limit for Raman spectrum plots in cm^-1.",
    )
    parser.add_argument(
        "--max-optical-modes",
        type=int,
        default=None,
        help="Optional debug limit on the number of optical modes processed per analysis.",
    )
    parser.add_argument(
        "--no-cueq",
        action="store_true",
        help="Disable CUDA equivariant kernels even on GPU.",
    )
    parser.add_argument(
        "--no-oeq",
        action="store_true",
        help="Disable OEQ kernels.",
    )
    return parser.parse_args()


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def axis_settings(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(top=False, bottom=True, left=True, right=False, width=1.2, length=5)
    ax.tick_params(which="minor", width=0.9, length=3)


def resolve_default_structure(model_basename: str, fallback: Path) -> Path:
    candidates = sorted(
        RUNS_ROOT.glob("SiO2-*/SiO2-mp-7000/dielectric_relax_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        model_path = summary.get("model_path", "")
        if Path(model_path).name != model_basename:
            continue
        relaxed = summary_path.parent / "relaxed_zero.annotated.extxyz"
        if relaxed.exists():
            return relaxed
    return fallback


def build_configs(args: argparse.Namespace) -> tuple[ModelAnalysisConfig, ModelAnalysisConfig]:
    direct_fallback = HERE / "SiO2.xyz"
    foundation_fallback = direct_fallback
    direct_structure = args.direct_structure or resolve_default_structure(args.direct_model.name, direct_fallback)
    foundation_structure = args.foundation_structure or resolve_default_structure(
        args.foundation_model.name, foundation_fallback
    )

    direct = ModelAnalysisConfig(
        label="direct",
        model_path=args.direct_model.resolve(),
        energy_head=args.direct_energy_head,
        response_head=args.direct_response_head,
        structure_path=direct_structure.resolve(),
        color=DIRECT_COLOR,
    )
    foundation = ModelAnalysisConfig(
        label="foundation",
        model_path=args.foundation_model.resolve(),
        energy_head=args.foundation_energy_head,
        response_head=args.foundation_response_head,
        structure_path=foundation_structure.resolve(),
        color=FOUNDATION_COLOR,
    )
    return direct, foundation


def build_calculator(model_path: Path, head: str, device: str, dtype: str, enable_cueq: bool, enable_oeq: bool):
    calc = MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype=dtype,
        model_type="MACEField",
        enable_cueq=enable_cueq,
        enable_oeq=enable_oeq,
        head=head,
    )
    calc.electric_field = np.zeros(3, dtype=float)
    return calc


def build_bundle(config: ModelAnalysisConfig, args: argparse.Namespace) -> CalculatorBundle:
    use_cueq = args.device.startswith("cuda") and not args.no_cueq
    common = {
        "device": args.device,
        "dtype": args.dtype,
        "enable_cueq": use_cueq,
        "enable_oeq": (not use_cueq and not args.no_oeq),
    }
    energy_calc = build_calculator(config.model_path, config.energy_head, **common)
    if config.response_head == config.energy_head:
        response_calc = energy_calc
    else:
        response_calc = build_calculator(config.model_path, config.response_head, **common)
    return CalculatorBundle(
        energy_calc=energy_calc,
        response_calc=response_calc,
    )


def coerce_hessian_to_cartesian_matrix(hessian_raw: np.ndarray, natoms: int) -> np.ndarray:
    hessian = np.asarray(hessian_raw, dtype=float)
    dim = 3 * natoms
    if hessian.shape == (dim, dim):
        return hessian
    if hessian.ndim == 4 and hessian.shape == (natoms, natoms, 3, 3):
        return np.transpose(hessian, (0, 2, 1, 3)).reshape(dim, dim)
    if hessian.ndim == 3 and hessian.shape == (dim, natoms, 3):
        return hessian.reshape(dim, dim)
    raise ValueError(f"Unsupported Hessian shape {hessian.shape}")


def compute_polarizability(calc: MACECalculator, atoms) -> np.ndarray:
    tensor = np.asarray(calc.get_property("polarizability", atoms), dtype=float).reshape(3, 3)
    return 0.5 * (tensor + tensor.T)


def compute_hessian(calc: MACECalculator, atoms) -> np.ndarray:
    hessian_raw = calc.get_hessian(atoms=atoms)
    hessian = coerce_hessian_to_cartesian_matrix(hessian_raw, natoms=len(atoms))
    return 0.5 * (hessian + hessian.T)


def mode_frequencies_and_vectors(hessian: np.ndarray, masses_amu: np.ndarray):
    mass_vector = np.repeat(np.asarray(masses_amu, dtype=float), 3)
    inv_sqrt_mass = 1.0 / np.sqrt(mass_vector)
    dynamical = inv_sqrt_mass[:, None] * hessian * inv_sqrt_mass[None, :]
    dynamical = 0.5 * (dynamical + dynamical.T)
    eigvals, eigvecs = np.linalg.eigh(dynamical)
    conversion = EV_J / (ANG_M**2 * AMU_KG)
    omega_sq_si = eigvals * conversion
    freq_cm = np.sign(omega_sq_si) * np.sqrt(np.abs(omega_sq_si)) / (2.0 * np.pi * LIGHT_CM_S)
    cart_mode_vectors = inv_sqrt_mass[:, None] * eigvecs
    return eigvals, freq_cm, eigvecs, cart_mode_vectors


def acoustic_and_optical_indices(freq_cm: np.ndarray) -> tuple[list[int], list[int]]:
    acoustic = np.argsort(np.abs(freq_cm))[:3].tolist()
    acoustic_set = set(acoustic)
    optical = [idx for idx in np.argsort(freq_cm) if idx not in acoustic_set]
    return acoustic, optical


def tensor_invariants(derivative_tensor: np.ndarray) -> dict[str, float]:
    tensor = 0.5 * (np.asarray(derivative_tensor, dtype=float) + np.asarray(derivative_tensor, dtype=float).T)
    iso = float(np.trace(tensor) / 3.0)
    xx, yy, zz = tensor[0, 0], tensor[1, 1], tensor[2, 2]
    xy, xz, yz = tensor[0, 1], tensor[0, 2], tensor[1, 2]
    gamma_sq = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
    gamma_sq += 3.0 * (xy**2 + xz**2 + yz**2)
    frob = float(np.linalg.norm(tensor))
    activity = 45.0 * iso**2 + 7.0 * gamma_sq
    return {
        "iso_derivative": iso,
        "anisotropy_sq": float(gamma_sq),
        "frobenius_norm": frob,
        "raman_activity": float(activity),
    }


def analyze_mode_derivatives(
    *,
    atoms,
    response_calc: MACECalculator,
    frequencies_cm: np.ndarray,
    cart_modes: np.ndarray,
    mode_indices: Iterable[int],
    delta_q: float,
    basis_label: str,
    response_label: str,
) -> list[dict[str, float]]:
    natoms = len(atoms)
    rows: list[dict[str, float]] = []
    for mode_idx in mode_indices:
        displacement = cart_modes[:, mode_idx].reshape(natoms, 3) * delta_q
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions = plus.positions + displacement
        minus.positions = minus.positions - displacement

        alpha_plus = compute_polarizability(response_calc, plus)
        alpha_minus = compute_polarizability(response_calc, minus)
        derivative = (alpha_plus - alpha_minus) / (2.0 * delta_q)
        derivative = 0.5 * (derivative + derivative.T)

        invariants = tensor_invariants(derivative)
        rows.append(
            {
                "basis_label": basis_label,
                "response_label": response_label,
                "mode_index": int(mode_idx),
                "frequency_cm-1": float(frequencies_cm[mode_idx]),
                "alpha_xx_prime": float(derivative[0, 0]),
                "alpha_yy_prime": float(derivative[1, 1]),
                "alpha_zz_prime": float(derivative[2, 2]),
                "alpha_xy_prime": float(derivative[0, 1]),
                "alpha_xz_prime": float(derivative[0, 2]),
                "alpha_yz_prime": float(derivative[1, 2]),
                "max_cartesian_displacement_A": float(np.max(np.linalg.norm(displacement, axis=1))),
                "rms_cartesian_displacement_A": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
                **invariants,
            }
        )
    return rows


def native_mode_summary(
    config: ModelAnalysisConfig,
    bundle: CalculatorBundle,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], list[int], list[int]]:
    atoms = read(config.structure_path)
    hessian = compute_hessian(bundle.energy_calc, atoms)
    eigvals, frequencies_cm, eigvecs_mw, cart_modes = mode_frequencies_and_vectors(hessian, atoms.get_masses())
    acoustic, optical = acoustic_and_optical_indices(frequencies_cm)
    if args.max_optical_modes is not None:
        optical = optical[: args.max_optical_modes]

    rows = analyze_mode_derivatives(
        atoms=atoms,
        response_calc=bundle.response_calc,
        frequencies_cm=frequencies_cm,
        cart_modes=cart_modes,
        mode_indices=optical,
        delta_q=args.delta_q,
        basis_label=config.label,
        response_label=config.label,
    )

    positive_optical = [row for row in rows if row["frequency_cm-1"] > 0.0]
    summary = {
        "label": config.label,
        "model_path": str(config.model_path),
        "energy_head": config.energy_head,
        "response_head": config.response_head,
        "structure_path": str(config.structure_path),
        "natoms": len(atoms),
        "n_modes_total": int(len(frequencies_cm)),
        "n_optical_modes_reported": int(len(rows)),
        "n_negative_optical_modes": int(sum(row["frequency_cm-1"] < 0.0 for row in rows)),
        "acoustic_mode_indices": acoustic,
        "min_optical_frequency_cm-1": float(np.min([row["frequency_cm-1"] for row in rows])) if rows else math.nan,
        "max_optical_frequency_cm-1": float(np.max([row["frequency_cm-1"] for row in rows])) if rows else math.nan,
        "mean_positive_optical_frequency_cm-1": (
            float(np.mean([row["frequency_cm-1"] for row in positive_optical])) if positive_optical else math.nan
        ),
        "total_raman_activity": float(np.sum([row["raman_activity"] for row in positive_optical])) if positive_optical else 0.0,
    }
    return summary, frequencies_cm, eigvecs_mw, cart_modes, rows, acoustic, optical


def common_basis_analysis(
    direct_config: ModelAnalysisConfig,
    direct_bundle: CalculatorBundle,
    foundation_bundle: CalculatorBundle,
    common_frequencies_cm: np.ndarray,
    common_cart_modes: np.ndarray,
    optical_mode_indices: list[int],
    args: argparse.Namespace,
):
    atoms = read(direct_config.structure_path)
    optical = list(optical_mode_indices)

    direct_rows = analyze_mode_derivatives(
        atoms=atoms,
        response_calc=direct_bundle.response_calc,
        frequencies_cm=common_frequencies_cm,
        cart_modes=common_cart_modes,
        mode_indices=optical,
        delta_q=args.delta_q,
        basis_label="direct_basis",
        response_label="direct_response",
    )
    foundation_rows = analyze_mode_derivatives(
        atoms=atoms,
        response_calc=foundation_bundle.response_calc,
        frequencies_cm=common_frequencies_cm,
        cart_modes=common_cart_modes,
        mode_indices=optical,
        delta_q=args.delta_q,
        basis_label="direct_basis",
        response_label="foundation_response",
    )

    paired_rows = []
    for direct_row, foundation_row in zip(direct_rows, foundation_rows, strict=True):
        ratio = safe_ratio(foundation_row["raman_activity"], direct_row["raman_activity"])
        paired_rows.append(
            {
                "mode_index": direct_row["mode_index"],
                "frequency_cm-1": direct_row["frequency_cm-1"],
                "direct_raman_activity": direct_row["raman_activity"],
                "foundation_raman_activity": foundation_row["raman_activity"],
                "activity_ratio_foundation_over_direct": ratio,
                "direct_frobenius_norm": direct_row["frobenius_norm"],
                "foundation_frobenius_norm": foundation_row["frobenius_norm"],
                "frobenius_ratio_foundation_over_direct": safe_ratio(
                    foundation_row["frobenius_norm"], direct_row["frobenius_norm"]
                ),
                "direct_iso_derivative": direct_row["iso_derivative"],
                "foundation_iso_derivative": foundation_row["iso_derivative"],
                "direct_anisotropy_sq": direct_row["anisotropy_sq"],
                "foundation_anisotropy_sq": foundation_row["anisotropy_sq"],
            }
        )

    direct_activities = np.asarray([row["direct_raman_activity"] for row in paired_rows], dtype=float)
    foundation_activities = np.asarray([row["foundation_raman_activity"] for row in paired_rows], dtype=float)
    direct_norms = np.asarray([row["direct_frobenius_norm"] for row in paired_rows], dtype=float)
    foundation_norms = np.asarray([row["foundation_frobenius_norm"] for row in paired_rows], dtype=float)
    frequencies = np.asarray([row["frequency_cm-1"] for row in paired_rows], dtype=float)

    summary = {
        "basis": "direct_native_modes",
        "structure_path": str(direct_config.structure_path),
        "n_modes": int(len(paired_rows)),
        "raman_activity_correlation": pearson_or_nan(direct_activities, foundation_activities),
        "frobenius_norm_correlation": pearson_or_nan(direct_norms, foundation_norms),
        "total_direct_raman_activity": float(np.sum(direct_activities)),
        "total_foundation_raman_activity": float(np.sum(foundation_activities)),
        "total_activity_ratio_foundation_over_direct": safe_ratio(np.sum(foundation_activities), np.sum(direct_activities)),
        "mean_foundation_over_direct_frequency_weighted_activity_ratio": weighted_mean_ratio(
            numerator=foundation_activities,
            denominator=direct_activities,
            weights=np.clip(frequencies, 0.0, None),
        ),
    }

    if paired_rows:
        direct_activity_values = np.asarray([row["direct_raman_activity"] for row in paired_rows], dtype=float)
        strength_threshold = float(np.percentile(direct_activity_values, 75.0))
        strong_modes = [row for row in paired_rows if row["direct_raman_activity"] >= strength_threshold]
        strong_modes = sorted(
            strong_modes,
            key=lambda row: (
                np.inf if not np.isfinite(row["activity_ratio_foundation_over_direct"]) else row["activity_ratio_foundation_over_direct"],
                -row["direct_raman_activity"],
            ),
        )
        summary["most_suppressed_strong_modes"] = strong_modes[:10]
    else:
        summary["most_suppressed_strong_modes"] = []
    return summary, direct_rows, foundation_rows, paired_rows


def safe_ratio(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if not np.isfinite(denom) or abs(denom) < 1.0e-16:
        return math.nan
    return float(numerator) / denom


def weighted_mean_ratio(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(numerator) & np.isfinite(denominator) & np.isfinite(weights) & (np.abs(denominator) > 1.0e-16)
    if not np.any(mask):
        return math.nan
    ratios = numerator[mask] / denominator[mask]
    return float(np.average(ratios, weights=weights[mask]))


def pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return math.nan
    x_sel = np.asarray(x[mask], dtype=float)
    y_sel = np.asarray(y[mask], dtype=float)
    if np.allclose(x_sel, x_sel[0]) or np.allclose(y_sel, y_sel[0]):
        return math.nan
    return float(np.corrcoef(x_sel, y_sel)[0, 1])


def match_native_modes(
    direct_rows: list[dict[str, float]],
    foundation_rows: list[dict[str, float]],
    direct_eigvecs_mw: np.ndarray,
    foundation_eigvecs_mw: np.ndarray,
) -> tuple[list[dict[str, float]], np.ndarray]:
    direct_indices = [int(row["mode_index"]) for row in direct_rows]
    foundation_indices = [int(row["mode_index"]) for row in foundation_rows]
    direct_modes = direct_eigvecs_mw[:, direct_indices]
    foundation_modes = foundation_eigvecs_mw[:, foundation_indices]
    overlap = np.abs(direct_modes.T @ foundation_modes)

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(1.0 - overlap)
    else:  # pragma: no cover
        row_idx, col_idx = greedy_assignment(overlap)

    direct_by_index = {int(row["mode_index"]): row for row in direct_rows}
    foundation_by_index = {int(row["mode_index"]): row for row in foundation_rows}

    matches = []
    for i, j in sorted(zip(row_idx, col_idx, strict=True), key=lambda pair: direct_indices[pair[0]]):
        direct_mode = direct_indices[i]
        foundation_mode = foundation_indices[j]
        direct_row = direct_by_index[direct_mode]
        foundation_row = foundation_by_index[foundation_mode]
        matches.append(
            {
                "direct_mode_index": direct_mode,
                "foundation_mode_index": foundation_mode,
                "overlap_abs": float(overlap[i, j]),
                "direct_frequency_cm-1": direct_row["frequency_cm-1"],
                "foundation_frequency_cm-1": foundation_row["frequency_cm-1"],
                "frequency_ratio_foundation_over_direct": safe_ratio(
                    foundation_row["frequency_cm-1"], direct_row["frequency_cm-1"]
                ),
                "direct_raman_activity": direct_row["raman_activity"],
                "foundation_raman_activity": foundation_row["raman_activity"],
                "activity_ratio_foundation_over_direct": safe_ratio(
                    foundation_row["raman_activity"], direct_row["raman_activity"]
                ),
                "direct_frobenius_norm": direct_row["frobenius_norm"],
                "foundation_frobenius_norm": foundation_row["frobenius_norm"],
                "frobenius_ratio_foundation_over_direct": safe_ratio(
                    foundation_row["frobenius_norm"], direct_row["frobenius_norm"]
                ),
            }
        )
    return matches, overlap


def greedy_assignment(overlap: np.ndarray):  # pragma: no cover
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    assignments = []
    flat = np.dstack(np.unravel_index(np.argsort(overlap.ravel())[::-1], overlap.shape))[0]
    for row, col in flat:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(int(row))
        used_cols.add(int(col))
        assignments.append((int(row), int(col)))
        if len(used_rows) == overlap.shape[0]:
            break
    row_idx = np.array([pair[0] for pair in assignments], dtype=int)
    col_idx = np.array([pair[1] for pair in assignments], dtype=int)
    return row_idx, col_idx


def gaussian_spectrum(rows: list[dict[str, float]], sigma_cm: float, max_frequency_cm: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, max_frequency_cm, 4000)
    spectrum = np.zeros_like(grid)
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    prefactor = 1.0 / (sigma_cm * np.sqrt(2.0 * np.pi))
    for row in rows:
        freq = float(row["frequency_cm-1"])
        activity = float(row["raman_activity"])
        if not np.isfinite(freq) or not np.isfinite(activity) or freq <= 0.0 or activity <= 0.0:
            continue
        spectrum += activity * prefactor * np.exp(-0.5 * ((grid - freq) / sigma_cm) ** 2)
    return grid, spectrum


def plot_raman_spectra(
    *,
    direct_native_rows: list[dict[str, float]],
    foundation_native_rows: list[dict[str, float]],
    foundation_common_rows: list[dict[str, float]],
    output_dir: Path,
    sigma_cm: float,
    max_frequency_cm: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    axis_settings(ax)

    grid, direct_curve = gaussian_spectrum(direct_native_rows, sigma_cm, max_frequency_cm)
    _, foundation_curve = gaussian_spectrum(foundation_native_rows, sigma_cm, max_frequency_cm)
    _, foundation_common_curve = gaussian_spectrum(foundation_common_rows, sigma_cm, max_frequency_cm)

    ax.plot(grid, direct_curve, color=DIRECT_COLOR, lw=2.0, label="Direct native")
    ax.plot(grid, foundation_curve, color=FOUNDATION_COLOR, lw=2.0, label="Foundation native")
    ax.plot(grid, foundation_common_curve, color=COMMON_COLOR, lw=2.0, label="Foundation on direct modes")

    ax.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax.set_ylabel("Broadened Raman Activity")
    ax.set_xlim(0.0, max_frequency_cm)
    ax.legend(frameon=False)

    path = output_dir / "sio2_mode_resolved_raman_spectra.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_common_basis_activity_parity(common_rows: list[dict[str, float]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    axis_settings(ax)

    direct = np.asarray([row["direct_raman_activity"] for row in common_rows], dtype=float)
    foundation = np.asarray([row["foundation_raman_activity"] for row in common_rows], dtype=float)
    freq = np.asarray([row["frequency_cm-1"] for row in common_rows], dtype=float)

    vmax = float(np.nanmax(np.concatenate([direct, foundation]))) if len(common_rows) else 1.0
    ax.scatter(direct, foundation, c=freq, cmap="viridis", s=22, alpha=0.9, linewidths=0.0)
    ax.plot([0.0, vmax], [0.0, vmax], color="0.35", lw=1.2, ls="--")
    ax.set_xlabel("Direct Raman Activity")
    ax.set_ylabel("Foundation Raman Activity\n(on direct-mode basis)")

    cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.02)
    cbar.set_label(r"Direct-mode frequency (cm$^{-1}$)")

    path = output_dir / "sio2_mode_resolved_common_basis_activity_parity.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_native_frequency_parity(native_matches: list[dict[str, float]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    axis_settings(ax)

    direct = np.asarray([row["direct_frequency_cm-1"] for row in native_matches], dtype=float)
    foundation = np.asarray([row["foundation_frequency_cm-1"] for row in native_matches], dtype=float)
    overlap = np.asarray([row["overlap_abs"] for row in native_matches], dtype=float)
    vmax = float(np.nanmax(np.concatenate([direct, foundation]))) if len(native_matches) else 1.0

    ax.scatter(direct, foundation, c=overlap, cmap="magma", s=22, alpha=0.9, linewidths=0.0)
    ax.plot([0.0, vmax], [0.0, vmax], color="0.35", lw=1.2, ls="--")
    ax.set_xlabel(r"Direct native frequency (cm$^{-1}$)")
    ax.set_ylabel(r"Foundation native frequency (cm$^{-1}$)")

    cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.02)
    cbar.set_label("Absolute mode overlap")

    path = output_dir / "sio2_mode_resolved_native_frequency_parity.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def relative_structure_difference(direct_structure: Path, foundation_structure: Path) -> dict[str, float]:
    direct_atoms = read(direct_structure)
    foundation_atoms = read(foundation_structure)
    delta = np.asarray(foundation_atoms.get_positions() - direct_atoms.get_positions(), dtype=float)
    return {
        "rms_cartesian_difference_A": float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
        "max_abs_cartesian_difference_A": float(np.max(np.abs(delta))),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    direct_config, foundation_config = build_configs(args)
    direct_bundle = build_bundle(direct_config, args)
    foundation_bundle = build_bundle(foundation_config, args)

    direct_summary, direct_freq, direct_vecs_mw, direct_cart_modes, direct_rows, _, direct_optical = native_mode_summary(
        direct_config, direct_bundle, args
    )
    foundation_summary, foundation_freq, foundation_vecs_mw, _, foundation_rows, _, _ = native_mode_summary(
        foundation_config, foundation_bundle, args
    )

    common_summary, common_direct_rows, common_foundation_rows, common_paired_rows = common_basis_analysis(
        direct_config,
        direct_bundle,
        foundation_bundle,
        common_cart_modes=direct_cart_modes,
        common_frequencies_cm=direct_freq,
        optical_mode_indices=direct_optical,
        args=args,
    )
    native_matches, overlap_matrix = match_native_modes(
        direct_rows=direct_rows,
        foundation_rows=foundation_rows,
        direct_eigvecs_mw=direct_vecs_mw,
        foundation_eigvecs_mw=foundation_vecs_mw,
    )

    native_match_summary = {
        "n_matches": int(len(native_matches)),
        "mean_overlap_abs": float(np.mean([row["overlap_abs"] for row in native_matches])) if native_matches else math.nan,
        "median_overlap_abs": float(np.median([row["overlap_abs"] for row in native_matches])) if native_matches else math.nan,
        "mean_frequency_ratio_foundation_over_direct": (
            float(np.mean([row["frequency_ratio_foundation_over_direct"] for row in native_matches if np.isfinite(row["frequency_ratio_foundation_over_direct"])]))
            if native_matches
            else math.nan
        ),
        "raman_activity_correlation": pearson_or_nan(
            np.asarray([row["direct_raman_activity"] for row in native_matches], dtype=float),
            np.asarray([row["foundation_raman_activity"] for row in native_matches], dtype=float),
        ),
    }

    paths = {
        "direct_native_csv": str(write_csv(direct_rows, args.output_dir / "sio2_mode_resolved_direct_native.csv")),
        "foundation_native_csv": str(write_csv(foundation_rows, args.output_dir / "sio2_mode_resolved_foundation_native.csv")),
        "common_basis_direct_csv": str(
            write_csv(common_direct_rows, args.output_dir / "sio2_mode_resolved_direct_common_basis.csv")
        ),
        "common_basis_foundation_csv": str(
            write_csv(common_foundation_rows, args.output_dir / "sio2_mode_resolved_foundation_common_basis.csv")
        ),
        "common_basis_paired_csv": str(
            write_csv(common_paired_rows, args.output_dir / "sio2_mode_resolved_common_basis_pairs.csv")
        ),
        "native_matches_csv": str(write_csv(native_matches, args.output_dir / "sio2_mode_resolved_native_matches.csv")),
    }

    np.savez_compressed(
        args.output_dir / "sio2_mode_resolved_native_overlap.npz",
        overlap_matrix=overlap_matrix,
        direct_mode_indices=np.asarray([row["mode_index"] for row in direct_rows], dtype=int),
        foundation_mode_indices=np.asarray([row["mode_index"] for row in foundation_rows], dtype=int),
        direct_frequencies_cm=np.asarray([row["frequency_cm-1"] for row in direct_rows], dtype=float),
        foundation_frequencies_cm=np.asarray([row["frequency_cm-1"] for row in foundation_rows], dtype=float),
    )
    paths["native_overlap_npz"] = str(args.output_dir / "sio2_mode_resolved_native_overlap.npz")

    paths["raman_spectra_plot"] = str(
        plot_raman_spectra(
            direct_native_rows=direct_rows,
            foundation_native_rows=foundation_rows,
            foundation_common_rows=common_foundation_rows,
            output_dir=args.output_dir,
            sigma_cm=args.spectrum_sigma_cm,
            max_frequency_cm=args.max_frequency_cm,
        )
    )
    paths["common_basis_activity_parity_plot"] = str(
        plot_common_basis_activity_parity(common_paired_rows, args.output_dir)
    )
    paths["native_frequency_parity_plot"] = str(plot_native_frequency_parity(native_matches, args.output_dir))

    summary = {
        "analysis_parameters": {
            "device": args.device,
            "dtype": args.dtype,
            "delta_q_sqrt_amu_A": args.delta_q,
            "spectrum_sigma_cm-1": args.spectrum_sigma_cm,
            "max_frequency_cm-1": args.max_frequency_cm,
            "max_optical_modes": args.max_optical_modes,
        },
        "structure_difference": relative_structure_difference(
            direct_config.structure_path, foundation_config.structure_path
        ),
        "direct_native": direct_summary,
        "foundation_native": foundation_summary,
        "common_basis": common_summary,
        "native_matching": native_match_summary,
        "paths": paths,
    }

    summary_path = args.output_dir / "sio2_mode_resolved_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    main()
