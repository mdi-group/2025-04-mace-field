#!/usr/bin/env python3
"""Spectroscopy analysis for annotated SiO2 MACEField trajectories.

This script is designed around the annotated `extxyz` trajectories produced by
the LAMMPS workflows in `scripts/LAMMPs`, e.g. `production.annotated.extxyz`.

It extracts polarization and polarizability time series, computes
autocorrelation-based IR / Raman spectra, estimates dielectric constants, and
saves compact analysis outputs that are easy to reuse in notebooks.

Examples
--------
Analyse the newest SiO2 annotated trajectory automatically:

    python Spectroscopy.py

Analyse an explicit trajectory and save to a custom prefix:

    python Spectroscopy.py \
      --input ../LAMMPs/MD/runs/SiO2-mp-7000-sc1x1x1-300K-200ps-.../SiO2-mp-7000/production.annotated.extxyz \
      --output-prefix analysis_outputs/sio2_run_01

Run headless without Raman and save PNGs:

    python Spectroscopy.py --no-raman --save-plots --no-show
"""

from __future__ import annotations

import argparse
import json
import csv
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


# Constants and unit conversions
kB = 8.617333262145e-5  # eV K^-1
eps0const = 5.5263499562e-3  # [e * Volt^-1 * Angstrom^-1]
THZ_TO_CM_INV = 33.35640951981521
DEFAULT_GAUSSIAN_BROADENING_CM = 20.0
SPECTRUM_COLOR = "#0055d4"
REFERENCE_COLOR = "0.2"
REFERENCE_FILL = "0.75"
REFERENCE_IR_PEAKS_CM = {
    "omega_1_cm-1": 1041.0,
    "omega_2_cm-1": 420.0,
    "omega_3_cm-1": 765.0,
}
DEFAULT_EXPERIMENTAL_RAMAN_URL = "https://www.geologie-lyon.fr/Raman/spectres/quartz.txt"
EXPERIMENTAL_RAMAN_PEAKS_CM = np.array(
    [128.0, 206.0, 264.0, 354.0, 390.0, 450.0, 464.0, 697.0, 796.0, 808.0, 1069.0, 1162.0],
    dtype=np.float64,
)
EXPERIMENTAL_RAMAN_COLOR = "black"
COMMON_MODE_COLOR = "#14805e"


def axis_settings(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(top=False, bottom=True, left=True, right=False, width=1.2, length=5)
    ax.tick_params(which="minor", width=0.9, length=3)


def hide_y_ticks_and_values(ax):
    """Hide all y-axis tick marks, tick labels, minor ticks, and offset text."""
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.yaxis.get_offset_text().set_visible(False)


def plot_init(label_x, label_y, title, figsize=(4.2, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    axis_settings(ax)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    if title:
        ax.set_title(title)
    return fig, ax


def gaussian(x, amplitude, mu, sigma):
    return amplitude / np.sqrt(2.0 * np.pi * sigma**2) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_kernel(x, sigma):
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)


def gaussian_broaden(freq_cm, data, sigma_cm, mode=0):
    if mode == 0:
        freq_sym = np.concatenate((-freq_cm[::-1][:-1], freq_cm))
        data_sym = np.concatenate((data[::-1][:-1], data))
        kernel = gaussian_kernel(freq_sym, sigma_cm)
        broadened = np.convolve(data_sym, kernel, mode="same")[len(data) - 1 : len(freq_sym)]
        return broadened

    broadened = np.zeros(len(freq_cm))
    for i, value in enumerate(data):
        gauss = gaussian(freq_cm, 1.0, freq_cm[i], sigma_cm)
        broadened += value * gauss / np.sum(gauss)
    return broadened


def ylim_range(data):
    ymax = float(np.nanmax(data))
    ymin = float(np.nanmin(data))
    pad = (ymax - ymin) * 0.15 if ymax != ymin else max(abs(ymax), 1.0) * 0.15
    return ymin - pad, ymax + pad


def set_lims(x, xi, xf, y):
    idx_i = np.abs(x - xi).argmin()
    idx_f = np.abs(x - xf).argmin()
    lo, hi = sorted((idx_i, idx_f))
    return ylim_range(y[lo : hi + 1])


def resolve_default_input() -> Path:
    script_dir = Path(__file__).resolve().parent
    runs_root = script_dir.parent / "LAMMPs" / "MD" / "runs"
    candidates = sorted(
        runs_root.glob("SiO2-*/SiO2-mp-7000/production.annotated.extxyz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find any SiO2 annotated trajectories under scripts/LAMMPs/MD/runs. "
            "Pass --input explicitly."
        )
    return candidates[0]


def normalize_curve(curve):
    curve = np.asarray(curve, dtype=np.float64)
    max_abs = np.nanmax(np.abs(curve))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        return curve
    return curve / max_abs


def positive_limits(values, pad_frac=0.08, floor=0.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor, floor + 1.0
    vmax = float(np.nanmax(values))
    if vmax <= floor:
        return floor, floor + 1.0
    return floor, vmax * (1.0 + pad_frac)


def nearest_peak_frequency(freq_cm, intensity, target_cm, window_cm=120.0):
    freq_cm = np.asarray(freq_cm, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    mask = np.isfinite(freq_cm) & np.isfinite(intensity) & (np.abs(freq_cm - target_cm) <= window_cm)
    if not np.any(mask):
        return np.nan

    freq_sel = freq_cm[mask]
    intensity_sel = intensity[mask]
    if freq_sel.size == 1:
        return float(freq_sel[0])

    peaks = []
    for i in range(1, len(freq_sel) - 1):
        if intensity_sel[i] >= intensity_sel[i - 1] and intensity_sel[i] >= intensity_sel[i + 1]:
            peaks.append(i)

    if not peaks:
        idx = int(np.argmax(intensity_sel))
        return float(freq_sel[idx])

    peak_freqs = freq_sel[peaks]
    peak_intensities = intensity_sel[peaks]
    order = np.lexsort((np.abs(peak_freqs - target_cm), -peak_intensities))
    return float(peak_freqs[order[0]])


def symmetrize_tensor_series(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    return 0.5 * (alpha + np.swapaxes(alpha, -1, -2))


def alpha_isotropic_series(alpha):
    alpha = symmetrize_tensor_series(alpha)
    return np.trace(alpha, axis1=1, axis2=2) / 3.0


def alpha_anisotropy_series(alpha):
    alpha = symmetrize_tensor_series(alpha)
    axx = alpha[:, 0, 0]
    ayy = alpha[:, 1, 1]
    azz = alpha[:, 2, 2]
    axy = alpha[:, 0, 1]
    axz = alpha[:, 0, 2]
    ayz = alpha[:, 1, 2]
    gamma_sq = 0.5 * ((axx - ayy) ** 2 + (ayy - azz) ** 2 + (azz - axx) ** 2)
    gamma_sq += 3.0 * (axy**2 + axz**2 + ayz**2)
    return gamma_sq


def find_first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return key
    return None


def axis_to_index(axis: str) -> int:
    lookup = {"x": 0, "y": 1, "z": 2, "a": 0, "b": 1, "c": 2}
    try:
        return lookup[axis.lower()]
    except KeyError as exc:
        raise ValueError(f"Axis must be one of x, y, z, a, b, c; got {axis!r}") from exc


def resolve_reference_root() -> Path:
    return Path(__file__).resolve().parent / "allegro-pol-reference"


def load_allegro_pol_quartz_reference():
    root = resolve_reference_root()
    ir_path = root / "DFPT" / "SiO2-IR-dfpt.dat"
    eps_re_path = root / "DFPT" / "SiO2-epsre-dfpt.dat"
    eps_im_path = root / "DFPT" / "SiO2-epsim-dfpt.dat"
    if not (ir_path.exists() and eps_re_path.exists() and eps_im_path.exists()):
        return None

    ir_data = np.loadtxt(ir_path, comments="#")
    eps_re_data = np.loadtxt(eps_re_path)
    eps_im_data = np.loadtxt(eps_im_path)
    if ir_data.ndim != 2 or ir_data.shape[1] < 4:
        return None
    if eps_re_data.ndim != 2 or eps_re_data.shape[1] < 2:
        return None
    if eps_im_data.ndim != 2 or eps_im_data.shape[1] < 2:
        return None

    return {
        "label": "DFPT",
        "ir_freq_cm": ir_data[:, 1],
        "ir_intensity": ir_data[:, 3],
        "eps_re_freq_cm": eps_re_data[:, 0],
        "eps_re": eps_re_data[:, 1],
        "eps_im_freq_cm": eps_im_data[:, 0],
        "eps_im": eps_im_data[:, 1],
    }


def _is_url(source):
    source = str(source)
    return source.startswith("http://") or source.startswith("https://")


def _parse_two_column_spectrum_text(text):
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(",", " ").replace(";", " ")
        values = []
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        if len(values) >= 2:
            rows.append((values[0], values[1]))
    if not rows:
        return None
    data = np.asarray(rows, dtype=np.float64)
    freq = data[:, 0]
    intensity = data[:, 1]
    mask = np.isfinite(freq) & np.isfinite(intensity)
    freq = freq[mask]
    intensity = intensity[mask]
    if freq.size < 2:
        return None
    order = np.argsort(freq)
    return freq[order], intensity[order]


def load_experimental_raman_curve(source=None, cache_dir=None, timeout_s=15.0):
    """Load the ENS-Lyon quartz Raman text spectrum or a user-supplied 2-column spectrum.

    Parameters
    ----------
    source
        URL or local path. If omitted, use the ENS-Lyon quartz text spectrum URL.
    cache_dir
        Optional directory for a cached copy named ``quartz_ens_lyon_raman.txt``.
        If URL download fails, the cached file is used when present.
    """
    source = DEFAULT_EXPERIMENTAL_RAMAN_URL if source is None else str(source)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "quartz_ens_lyon_raman.txt"

    text = None
    if _is_url(source):
        try:
            with urllib.request.urlopen(source, timeout=timeout_s) as response:
                text = response.read().decode("utf-8", errors="replace")
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
        except (urllib.error.URLError, TimeoutError, OSError):
            if cache_path is not None and cache_path.exists():
                text = cache_path.read_text(encoding="utf-8", errors="replace")
            else:
                return None
    else:
        path = Path(source).expanduser()
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")

    return _parse_two_column_spectrum_text(text)


def load_experimental_raman_peaks(default_root=None):
    if default_root is None:
        default_root = Path(__file__).resolve().parent
    candidates = [
        Path(default_root) / "alpha_sio2_experimental_peaks_ens_lyon.csv",
        Path.cwd() / "alpha_sio2_experimental_peaks_ens_lyon.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = np.genfromtxt(path, delimiter=",", names=True)
                values = np.atleast_1d(data["frequency_cm-1"]).astype(np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    return values
            except Exception:
                continue
    return EXPERIMENTAL_RAMAN_PEAKS_CM.copy()


def normalize_experimental_curve(freq, intensity, freq_min, freq_max, target_max=1.0):
    freq = np.asarray(freq, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    mask = np.isfinite(freq) & np.isfinite(intensity) & (freq >= freq_min) & (freq <= freq_max)
    if not np.any(mask):
        return freq, intensity * 0.0
    y = intensity.copy()
    baseline = float(np.nanpercentile(y[mask], 2.0))
    y = y - baseline
    y[y < 0.0] = 0.0
    ymax = float(np.nanmax(y[mask]))
    if not np.isfinite(ymax) or ymax <= 0.0:
        return freq, y * 0.0
    return freq, y / ymax * float(target_max)


def gaussian_smooth_uniform_grid(grid, values, sigma_cm):
    grid = np.asarray(grid, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if sigma_cm is None or sigma_cm <= 0.0 or grid.size < 2:
        return values.copy()

    dx = float(np.median(np.diff(grid)))
    if not np.isfinite(dx) or dx <= 0.0:
        return values.copy()

    half_width = max(1, int(np.ceil(6.0 * float(sigma_cm) / dx)))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64) * dx
    kernel = np.exp(-0.5 * (offsets / float(sigma_cm)) ** 2)
    kernel_sum = float(np.sum(kernel))
    if not np.isfinite(kernel_sum) or kernel_sum <= 0.0:
        return values.copy()
    kernel /= kernel_sum
    return np.convolve(values, kernel, mode="same")


def prepare_experimental_mode_resolved_curve(
    freq,
    intensity,
    *,
    freq_min,
    freq_max,
    sigma_cm,
    target_max=1.0,
    n_grid=4000,
):
    freq = np.asarray(freq, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    finite = np.isfinite(freq) & np.isfinite(intensity)
    if not np.any(finite):
        grid = np.linspace(freq_min, freq_max, n_grid)
        return grid, np.zeros_like(grid)

    freq = freq[finite]
    intensity = intensity[finite]
    order = np.argsort(freq)
    freq = freq[order]
    intensity = intensity[order]

    window = (freq >= freq_min) & (freq <= freq_max)
    if not np.any(window):
        grid = np.linspace(freq_min, freq_max, n_grid)
        return grid, np.zeros_like(grid)

    y = intensity.copy()
    baseline = float(np.nanpercentile(y[window], 2.0))
    y = y - baseline
    y[y < 0.0] = 0.0

    grid = np.linspace(freq_min, freq_max, n_grid)
    y_grid = np.interp(grid, freq, y, left=0.0, right=0.0)
    y_grid = gaussian_smooth_uniform_grid(grid, y_grid, sigma_cm)

    ymax = float(np.nanmax(y_grid)) if y_grid.size else 0.0
    if not np.isfinite(ymax) or ymax <= 0.0:
        return grid, np.zeros_like(grid)
    return grid, y_grid / ymax * float(target_max)


def load_mode_resolved_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def load_mode_resolved_results(mode_resolved_dir):
    if mode_resolved_dir is None:
        return None
    mode_resolved_dir = Path(mode_resolved_dir)
    direct_rows = load_mode_resolved_csv(mode_resolved_dir / "sio2_mode_resolved_direct_native.csv")
    foundation_rows = load_mode_resolved_csv(mode_resolved_dir / "sio2_mode_resolved_foundation_native.csv")
    common_rows = load_mode_resolved_csv(mode_resolved_dir / "sio2_mode_resolved_foundation_common_basis.csv")
    if not direct_rows and not foundation_rows and not common_rows:
        return None
    return {
        "dir": str(mode_resolved_dir),
        "direct_native_rows": direct_rows,
        "foundation_native_rows": foundation_rows,
        "foundation_common_rows": common_rows,
    }


def infer_mode_resolved_dir(explicit_dir=None):
    if explicit_dir is not None:
        return Path(explicit_dir).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "plots" / "mode_resolved_polarizability",
        script_dir / "mode_resolved_polarizability",
        Path.cwd() / "mode_resolved_polarizability",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def gaussian_mode_resolved_spectrum(rows, sigma_cm, freq_min, freq_max):
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    grid = np.linspace(freq_min, freq_max, 4000)
    spectrum = np.zeros_like(grid)
    prefactor = 1.0 / (sigma_cm * np.sqrt(2.0 * np.pi))
    for row in rows:
        try:
            freq = float(row["frequency_cm-1"])
            activity = float(row["raman_activity"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(freq) or not np.isfinite(activity) or freq <= 0.0 or activity <= 0.0:
            continue
        spectrum += activity * prefactor * np.exp(-0.5 * ((grid - freq) / sigma_cm) ** 2)
    return grid, spectrum


def build_mode_resolved_series(payload, curves):
    if payload is None:
        return []
    cmap = plt.get_cmap("tab10")
    color_by_label = {curve.label: cmap(idx % 10) for idx, curve in enumerate(curves)}
    labels_lower = {curve.label.lower(): curve.label for curve in curves}
    foundation_label = None
    direct_label = None
    for low, label in labels_lower.items():
        if foundation_label is None and ("omat" in low or "foundation" in low):
            foundation_label = label
        if direct_label is None and "direct" in low:
            direct_label = label
    if foundation_label is None and curves:
        foundation_label = curves[0].label
    if direct_label is None and len(curves) > 1:
        direct_label = curves[1].label

    series = []
    if payload.get("foundation_native_rows"):
        series.append({
            "label": foundation_label or "foundation",
            "rows": payload["foundation_native_rows"],
            "color": color_by_label.get(foundation_label, cmap(0)),
            "linestyle": "-",
        })
    if payload.get("direct_native_rows"):
        series.append({
            "label": direct_label or "direct",
            "rows": payload["direct_native_rows"],
            "color": color_by_label.get(direct_label, cmap(1)),
            "linestyle": "-",
        })
    if payload.get("foundation_common_rows"):
        series.append({
            "label": f"{foundation_label or 'foundation'} on direct modes",
            "rows": payload["foundation_common_rows"],
            "color": COMMON_MODE_COLOR,
            "linestyle": "--",
        })
    return series


def plot_mode_resolved_subplot(
    ax,
    payload,
    curves,
    *,
    freq_min,
    freq_max,
    sigma_cm,
    experimental_curve=None,
    experimental_peaks=None,
):
    axis_settings(ax)
    ax.set_title("Mode-resolved Raman")
    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    hide_y_ticks_and_values(ax)

    series = build_mode_resolved_series(payload, curves)
    if not series:
        ax.text(0.5, 0.5, "Mode-resolved Raman data unavailable", ha="center", va="center", transform=ax.transAxes)
        return

    all_curves = []
    handles = []
    max_y = 0.0
    for item in series:
        grid, curve = gaussian_mode_resolved_spectrum(item["rows"], sigma_cm, freq_min, freq_max)
        ax.plot(grid, curve, lw=1.8, color=item["color"], ls=item["linestyle"], label=item["label"])
        ax.fill_between(grid, 0.0, curve, color=item["color"], alpha=0.16)
        handles.append(Line2D([0], [0], color=item["color"], lw=1.8, ls=item["linestyle"], label=item["label"]))
        all_curves.append(curve)
        if np.any(np.isfinite(curve)):
            max_y = max(max_y, float(np.nanmax(curve)))

    if experimental_curve is not None and max_y > 0.0:
        exp_freq, exp_intensity = experimental_curve
        exp_grid, exp_scaled = prepare_experimental_mode_resolved_curve(
            exp_freq,
            exp_intensity,
            freq_min=freq_min,
            freq_max=freq_max,
            sigma_cm=sigma_cm,
            target_max=max_y,
        )
        ax.plot(
            exp_grid,
            exp_scaled,
            color=EXPERIMENTAL_RAMAN_COLOR,
            lw=2.0,
            ls=":",
            label="Exp. Raman",
            zorder=60,
        )
        handles.append(Line2D([0], [0], color=EXPERIMENTAL_RAMAN_COLOR, lw=2.0, ls=":", label="Exp. Raman"))
        all_curves.append(exp_scaled)
    if experimental_peaks is not None and len(experimental_peaks) and max_y > 0.0:
        peak_height = max_y * 0.22
        ax.vlines(experimental_peaks, 0.0, peak_height, color=EXPERIMENTAL_RAMAN_COLOR, lw=1.9, ls=":", alpha=0.90, zorder=61)
        ax.scatter(experimental_peaks, np.full_like(experimental_peaks, peak_height), marker="v", s=26, color=EXPERIMENTAL_RAMAN_COLOR, alpha=0.95, zorder=62)
        handles.append(Line2D([0], [0], color=EXPERIMENTAL_RAMAN_COLOR, lw=1.9, ls=":", marker="v", markersize=5, label="Exp. peaks"))

    if all_curves:
        concat = np.concatenate(all_curves)
        ax.set_ylim(*positive_limits(concat[np.isfinite(concat)], pad_frac=0.06))
        hide_y_ticks_and_values(ax)
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=10)


class Spectroscopy:
    """Spectroscopy analysis for annotated extxyz trajectories."""

    def __init__(
        self,
        trajectory_path,
        temp_k=None,
        freq_min_cm=0.0,
        freq_max_cm=1400.0,
        do_ir=True,
        do_raman=True,
        timestep_fs=None,
        autocorr_ensembles=1,
        backend="auto",
        save_prefix=None,
        gaussian_broadening_cm=DEFAULT_GAUSSIAN_BROADENING_CM,
        damp_frequency_tail=True,
        polar_axis="z",
    ):
        self.trajectory_path = Path(trajectory_path).resolve()
        self.temp = temp_k
        self.ωi = float(freq_min_cm)
        self.ωf = float(freq_max_cm)
        self.do_IR = bool(do_ir)
        self.do_Raman = bool(do_raman)
        self.timestep_fs = None if timestep_fs is None else float(timestep_fs)
        self.autocorr_ensembles = int(autocorr_ensembles)
        self.backend = self._resolve_backend(backend)
        self.save_prefix = None if save_prefix is None else Path(save_prefix)
        self.gaussian_broadening_cm = float(gaussian_broadening_cm)
        self.damp_frequency_tail = bool(damp_frequency_tail)
        self.polar_axis_label = polar_axis
        self.polar_axis = axis_to_index(polar_axis)

        self.parse_trajectory(self.trajectory_path)

        if self.do_IR:
            self.get_autocorr_P()
            self.get_IR()
        if self.do_Raman:
            self.get_autocorr_alpha()
            self.get_Raman()

        self.get_epsilon_inf_epsilon_0()
        if self.do_IR:
            self.get_epsilon_omega()
            self.get_reference_ir_peaks()

        if self.save_prefix:
            self.save_results(self.save_prefix)

    def _parallel_perpendicular(self, values):
        values = np.asarray(values, dtype=np.float64)
        parallel = float(values[self.polar_axis])
        perp_indices = [i for i in range(3) if i != self.polar_axis]
        perpendicular = float(np.mean(values[perp_indices]))
        return parallel, perpendicular

    def _resolve_backend(self, backend):
        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError("backend must be one of: auto, numpy, torch")
        if backend == "numpy":
            return "numpy"
        if backend == "torch":
            if torch is None:
                raise ImportError("backend='torch' requested but PyTorch is not installed")
            return "torch"
        if torch is not None and torch.cuda.is_available():
            return "torch"
        return "numpy"

    def _to_numpy(self, array):
        if torch is not None and isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def _fft_autocorr_backend(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = values.shape[0]
        n_fft = 1 << (2 * n - 1).bit_length()

        if self.backend == "torch":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            x = torch.as_tensor(values, dtype=torch.float64, device=device)
            spectrum = torch.fft.rfft(x, n=n_fft)
            autocorr = torch.fft.irfft(spectrum.conj() * spectrum, n=n_fft)[:n]
            return self._to_numpy(autocorr / n)

        spectrum = np.fft.rfft(values, n=n_fft)
        autocorr = np.fft.irfft(spectrum.conj() * spectrum, n=n_fft)[:n]
        return np.asarray(autocorr / n, dtype=np.float64)

    def unit_pol(self, polarization):
        pol_mod_frac = np.dot(np.linalg.inv(self.g), self.cell).diagonal()
        pol_frac = np.dot(self.g, polarization.T).T
        pol_wrapped = pol_frac % (np.sign(pol_frac) * pol_mod_frac)
        pol_wrapped = np.where(pol_wrapped > 0.5 * pol_mod_frac, pol_wrapped - pol_mod_frac, pol_wrapped)
        pol_wrapped = np.where(pol_wrapped < -0.5 * pol_mod_frac, pol_wrapped + pol_mod_frac, pol_wrapped)
        return np.dot(np.linalg.inv(self.g), pol_wrapped.T).T

    def get_metric(self):
        self.g = np.zeros((3, 3))
        self.g[:, 0] = self.cell[:, 0] / np.linalg.norm(self.cell[:, 0])
        self.g[:, 1] = self.cell[:, 1] / np.linalg.norm(self.cell[:, 1])
        self.g[:, 2] = self.cell[:, 2] / np.linalg.norm(self.cell[:, 2])

    def parse_trajectory(self, trajectory_path):
        data = read(str(trajectory_path), ":")
        if not data:
            raise ValueError(f"No frames found in {trajectory_path}")

        first = data[0]
        pol_key = find_first_present(first.info, ["MACE_polarization", "REF_polarization"])
        alpha_key = find_first_present(first.info, ["MACE_polarizability", "REF_polarizability"])
        bec_key = find_first_present(first.arrays, ["MACE_becs", "REF_becs"])
        if pol_key is None:
            raise KeyError(f"Could not find polarization in {trajectory_path.name}")
        if alpha_key is None:
            raise KeyError(f"Could not find polarizability in {trajectory_path.name}")
        if bec_key is None:
            raise KeyError(f"Could not find BECs in {trajectory_path.name}")

        self.polarization_key = pol_key
        self.polarizability_key = alpha_key
        self.bec_key = bec_key

        self.V = first.get_volume()
        self.n_t = len(data)
        self.cell = np.asarray(first.get_cell())

        time_ps = [frame.info.get("time_ps") for frame in data]
        if all(value is not None for value in time_ps):
            self.time = np.asarray(time_ps, dtype=np.float64)
            if self.timestep_fs is None and len(self.time) > 1:
                dt_ps = float(np.median(np.diff(self.time)))
                self.timestep_fs = dt_ps * 1e3
        else:
            if self.timestep_fs is None:
                timestep_token = first.info.get("timestep")
                if timestep_token is not None and len(data) > 1:
                    step_delta = data[1].info.get("timestep", timestep_token + 1) - timestep_token
                    if step_delta != 0:
                        time_delta_ps = data[1].info.get("time_ps")
                        if time_delta_ps is not None and first.info.get("time_ps") is not None:
                            self.timestep_fs = (time_delta_ps - first.info["time_ps"]) * 1e3 / step_delta
                if self.timestep_fs is None:
                    raise ValueError(
                        "Could not infer timestep from trajectory metadata. Pass --timestep-fs explicitly."
                    )
            timestep_ps = self.timestep_fs * 1e-3
            self.time = np.arange(self.n_t, dtype=np.float64) * timestep_ps

        if self.temp is None:
            temps = [frame.info.get("temperature_K") for frame in data if frame.info.get("temperature_K") is not None]
            if temps:
                self.temp = float(np.mean(temps))
            else:
                raise ValueError(
                    "Could not infer trajectory temperature from frame metadata. Pass --temperature-k explicitly."
                )

        self.E = np.asarray(
            [frame.info.get("lammps_total_energy", frame.info.get("REF_energy", np.nan)) for frame in data],
            dtype=np.float64,
        )
        self.P = np.asarray([frame.info[self.polarization_key] for frame in data], dtype=np.float64).reshape(self.n_t, 3)
        self.Z = np.asarray([frame.arrays[self.bec_key] for frame in data], dtype=np.float64)
        self.alpha = np.asarray(
            [frame.info[self.polarizability_key] for frame in data],
            dtype=np.float64,
        ).reshape(self.n_t, 3, 3)
        self.alpha = symmetrize_tensor_series(self.alpha)

        self.get_metric()
        self.P = self.unit_pol(self.P)

        self.n_ω = int(self.n_t / 2) + 1
        self.var_P = np.var(self.P, axis=0, dtype=np.float64)
        self.var_alpha = np.array([np.var(self.alpha[:, i, i], dtype=np.float64) for i in range(3)])
        self.alpha_iso = alpha_isotropic_series(self.alpha)
        self.alpha_aniso = alpha_anisotropy_series(self.alpha)
        self.var_alpha_iso = float(np.var(self.alpha_iso, dtype=np.float64))
        self.var_alpha_aniso = float(np.var(self.alpha_aniso, dtype=np.float64))
        self.P_avg = np.mean(self.P, axis=0)
        self.alpha_avg = np.array([np.mean(self.alpha[:, i, i]) for i in range(3)])

    def get_autocorr_mlmd(self, t, values, m=1):
        n = len(values)
        if n < 2:
            raise ValueError("Need at least two frames for autocorrelation analysis")
        dt_cm = (t[1] - t[0]) / THZ_TO_CM_INV
        centered = np.asarray(values - np.mean(values), dtype=np.float64)

        ω = np.fft.rfftfreq(n, t[1] - t[0]) * THZ_TO_CM_INV
        AF_t = np.zeros(n)
        AF_ω = None
        for i in range(1, m + 1):
            stop = int(i * n / m)
            values_aux = np.zeros(n, dtype=np.float64)
            values_aux[:stop] = centered[:stop]
            AF_t_m = self._fft_autocorr_backend(values_aux)
            AF_ω_m = np.fft.rfft(AF_t_m) * dt_cm
            AF_t += AF_t_m
            AF_ω = AF_ω_m if AF_ω is None else AF_ω + AF_ω_m

        AF_t /= m
        AF_ω /= m
        AF_ω *= np.pi
        return ω, AF_t, AF_ω

    def get_autocorr_P(self):
        self.P_AF_t = np.zeros((self.n_t, 3))
        self.P_AF_ω = np.zeros((self.n_ω, 3), dtype=complex)
        for i in range(3):
            self.ω, self.P_AF_t[:, i], self.P_AF_ω[:, i] = self.get_autocorr_mlmd(
                self.time, self.P[:, i], m=self.autocorr_ensembles
            )
            if self.damp_frequency_tail:
                imag_part = np.array(self.P_AF_ω[:, i].imag)
                real_part = np.array(self.P_AF_ω[:, i].real)
                vmax = np.max(np.abs(real_part))
                if vmax > 0:
                    real_part[np.abs(real_part) < 0.01 * vmax] = 0.0
                self.P_AF_ω[:, i] = real_part + 1.0j * imag_part

        denom = np.sum(self.var_P)
        self.P_AF_t_av = np.sum(self.P_AF_t, axis=1) / denom
        self.P_AF_ω_av = -np.sum(self.P_AF_ω, axis=1) / denom

    def get_autocorr_alpha(self):
        self.alpha_AF_t = np.zeros((self.n_t, 3))
        self.alpha_AF_ω = np.zeros((self.n_ω, 3), dtype=complex)
        for i in range(3):
            self.ω, self.alpha_AF_t[:, i], self.alpha_AF_ω[:, i] = self.get_autocorr_mlmd(
                self.time, self.alpha[:, i, i], m=self.autocorr_ensembles
            )
        denom = np.sum(self.var_alpha)
        self.alpha_AF_t_av = np.sum(self.alpha_AF_t, axis=1) / denom
        self.alpha_AF_ω_av = np.sum(self.alpha_AF_ω, axis=1) / denom
        self.ω, self.alpha_iso_AF_t, self.alpha_iso_AF_ω = self.get_autocorr_mlmd(
            self.time, self.alpha_iso, m=self.autocorr_ensembles
        )
        self.ω, self.alpha_aniso_AF_t, self.alpha_aniso_AF_ω = self.get_autocorr_mlmd(
            self.time, self.alpha_aniso, m=self.autocorr_ensembles
        )

    def get_IR(self):
        self.IR = self.P_AF_ω_av.real / np.sum(self.var_P) * self.ω**2

    def get_Raman(self):
        iso_norm = self.var_alpha_iso if np.isfinite(self.var_alpha_iso) and self.var_alpha_iso > 0.0 else 1.0
        aniso_norm = (
            self.var_alpha_aniso
            if np.isfinite(self.var_alpha_aniso) and self.var_alpha_aniso > 0.0
            else 1.0
        )
        raman_response = 45.0 * (self.alpha_iso_AF_ω.real / iso_norm)
        raman_response += 7.0 * (self.alpha_aniso_AF_ω.real / aniso_norm)
        self.Raman = np.abs(raman_response) * self.ω**2

    def get_epsilon_inf_epsilon_0(self):
        self.epsilon_inf = np.zeros(3)
        self.epsilon_0 = np.zeros(3)
        self.epsilon_ion = np.zeros(3)
        for i in range(3):
            self.epsilon_inf[i] = 1.0 + np.mean(self.alpha[:, i, i])
            self.epsilon_ion[i] = self.V * self.var_P[i] / (kB * self.temp * eps0const)
            self.epsilon_0[i] = self.epsilon_inf[i] + self.epsilon_ion[i]

    def get_epsilon_omega(self):
        self.epsilon_omega = np.zeros((self.n_ω, 3), dtype=complex)
        for i in range(3):
            self.epsilon_omega[:, i] = 1 + (self.epsilon_0[i] - 1) * (
                1 - 1j * self.ω * self.P_AF_ω[:, i] / self.var_P[i]
            )
        self.epsilon_omega_avg_real = np.mean(self.epsilon_omega.real, axis=1)
        self.epsilon_omega_avg_imag = np.mean(self.epsilon_omega.imag, axis=1)

    def get_reference_ir_peaks(self):
        # Use the same broadened IR magnitude that is actually visualized in the
        # comparison plots so reported peaks match the plotted spectra.
        ir_curve = np.abs(gaussian_broaden(self.ω, self.IR, self.gaussian_broadening_cm))
        self.reference_ir_peaks_cm = {
            key: nearest_peak_frequency(self.ω, ir_curve, target_cm=value)
            for key, value in REFERENCE_IR_PEAKS_CM.items()
        }

    def get_serializable_results(self):
        results = {
            "time_ps": self.time,
            "frequency_cm-1": self.ω if hasattr(self, "ω") else np.array([]),
            "energy_eV": self.E,
            "polarization": self.P,
            "becs": self.Z,
            "polarizability": self.alpha,
            "epsilon_inf": self.epsilon_inf,
            "epsilon_0": self.epsilon_0,
        }
        if self.do_IR:
            results.update(
                {
                    "P_autocorr_time": self.P_AF_t,
                    "P_autocorr_time_avg": self.P_AF_t_av,
                    "P_autocorr_freq_real": self.P_AF_ω.real,
                    "P_autocorr_freq_imag": self.P_AF_ω.imag,
                    "P_autocorr_freq_avg_real": self.P_AF_ω_av.real,
                    "P_autocorr_freq_avg_imag": self.P_AF_ω_av.imag,
                    "IR": self.IR,
                    "epsilon_omega_real": self.epsilon_omega.real,
                    "epsilon_omega_imag": self.epsilon_omega.imag,
                    "epsilon_omega_avg_real": self.epsilon_omega_avg_real,
                    "epsilon_omega_avg_imag": self.epsilon_omega_avg_imag,
                }
            )
        if self.do_Raman:
            results.update(
                {
                    "alpha_autocorr_time": self.alpha_AF_t,
                    "alpha_autocorr_time_avg": self.alpha_AF_t_av,
                    "alpha_autocorr_freq_real": self.alpha_AF_ω.real,
                    "alpha_autocorr_freq_imag": self.alpha_AF_ω.imag,
                    "alpha_autocorr_freq_avg_real": self.alpha_AF_ω_av.real,
                    "alpha_autocorr_freq_avg_imag": self.alpha_AF_ω_av.imag,
                    "alpha_iso_autocorr_time": self.alpha_iso_AF_t,
                    "alpha_iso_autocorr_freq_real": self.alpha_iso_AF_ω.real,
                    "alpha_iso_autocorr_freq_imag": self.alpha_iso_AF_ω.imag,
                    "alpha_aniso_autocorr_time": self.alpha_aniso_AF_t,
                    "alpha_aniso_autocorr_freq_real": self.alpha_aniso_AF_ω.real,
                    "alpha_aniso_autocorr_freq_imag": self.alpha_aniso_AF_ω.imag,
                    "Raman": self.Raman,
                }
            )
        return results

    def save_results(self, prefix):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        results = self.get_serializable_results()
        np.savez_compressed(prefix.with_suffix(".npz"), **results)

        summary = {
            "trajectory": str(self.trajectory_path),
            "backend": self.backend,
            "timestep_fs": self.timestep_fs,
            "autocorr_ensembles": self.autocorr_ensembles,
            "n_frames": self.n_t,
            "n_frequencies": self.n_ω,
            "temperature_K": self.temp,
            "cell_volume_A3": self.V,
            "polarization_key": self.polarization_key,
            "polarizability_key": self.polarizability_key,
            "bec_key": self.bec_key,
            "polar_axis": self.polar_axis_label,
            "epsilon_inf_avg": float(np.mean(self.epsilon_inf)),
            "epsilon_inf_parallel": self._parallel_perpendicular(self.epsilon_inf)[0],
            "epsilon_inf_perpendicular": self._parallel_perpendicular(self.epsilon_inf)[1],
            "epsilon_ion_avg": float(np.mean(self.epsilon_ion)),
            "epsilon_ion_parallel": self._parallel_perpendicular(self.epsilon_ion)[0],
            "epsilon_ion_perpendicular": self._parallel_perpendicular(self.epsilon_ion)[1],
            "epsilon_0_avg": float(np.mean(self.epsilon_0)),
            "epsilon_0_parallel": self._parallel_perpendicular(self.epsilon_0)[0],
            "epsilon_0_perpendicular": self._parallel_perpendicular(self.epsilon_0)[1],
        }
        if self.do_IR and hasattr(self, "reference_ir_peaks_cm"):
            summary["reference_peak_source"] = "broadened_IR_magnitude"
            summary.update(self.reference_ir_peaks_cm)
        with prefix.with_name(prefix.name + "_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        spectral_columns = [self.ω]
        spectral_header = ["frequency_cm-1"]
        if self.do_IR:
            spectral_columns.extend([self.IR, self.epsilon_omega_avg_real, self.epsilon_omega_avg_imag])
            spectral_header.extend(["IR", "epsilon_avg_real", "epsilon_avg_imag"])
        if self.do_Raman:
            spectral_columns.append(self.Raman)
            spectral_header.append("Raman")
        np.savetxt(
            prefix.with_name(prefix.name + "_spectra.csv"),
            np.column_stack(spectral_columns),
            delimiter=",",
            header=",".join(spectral_header),
            comments="",
        )

        corr_columns = [self.time]
        corr_header = ["time_ps"]
        if self.do_IR:
            corr_columns.append(self.P_AF_t_av)
            corr_header.append("P_autocorr_avg")
        if self.do_Raman:
            corr_columns.append(self.alpha_AF_t_av)
            corr_header.append("alpha_autocorr_avg")
        np.savetxt(
            prefix.with_name(prefix.name + "_correlation.csv"),
            np.column_stack(corr_columns),
            delimiter=",",
            header=",".join(corr_header),
            comments="",
        )

    def plot_autocorr_P(self, show=True, save_path=None):
        fig, ax = plot_init("$t$ (ps)", "", r"$\langle P(t)\cdot P(0)\rangle / \mathrm{var}(P)$")
        ax.plot(self.time, self.P_AF_t_av, lw=1.0, c=SPECTRUM_COLOR)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_autocorr_alpha(self, show=True, save_path=None):
        fig, ax = plot_init("$t$ (ps)", "", r"$\langle \alpha(t)\cdot \alpha(0)\rangle / \mathrm{var}(\alpha)$")
        ax.plot(self.time, self.alpha_AF_t_av, lw=1.0, c=SPECTRUM_COLOR)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_IR(self, do_smear=True, show=True, save_path=None):
        fig, ax = plot_init(r"$\omega$ (cm$^{-1}$)", "Infrared Intensity", "")
        ax.set_xlim(self.ωi, self.ωf)
        hide_y_ticks_and_values(ax)

        curve = gaussian_broaden(self.ω, self.IR, self.gaussian_broadening_cm) if do_smear else self.IR
        ax.plot(self.ω, curve, lw=1.0, c=SPECTRUM_COLOR, label="MLMD")
        ax.set_ylim(set_lims(self.ω, self.ωi, self.ωf, curve))
        hide_y_ticks_and_values(ax)

        ax.legend(frameon=False)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_Raman(self, do_smear=True, show=True, save_path=None):
        fig, ax = plot_init(r"$\omega$ (cm$^{-1}$)", "Raman Intensity", "")
        ax.set_xlim(self.ωi, self.ωf)
        hide_y_ticks_and_values(ax)

        curve = gaussian_broaden(self.ω, self.Raman, self.gaussian_broadening_cm) if do_smear else self.Raman
        ax.plot(self.ω, curve, lw=1.0, c=SPECTRUM_COLOR, label="MLMD")
        ax.set_ylim(set_lims(self.ω, self.ωi, self.ωf, curve))
        hide_y_ticks_and_values(ax)

        ax.legend(frameon=False)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_epsilon(self, do_smear=True, show=True, save_prefix=None):
        re_curve = (
            gaussian_broaden(self.ω, self.epsilon_omega_avg_real, self.gaussian_broadening_cm)
            if do_smear
            else self.epsilon_omega_avg_real
        )
        im_curve = (
            gaussian_broaden(self.ω, self.epsilon_omega_avg_imag, self.gaussian_broadening_cm)
            if do_smear
            else self.epsilon_omega_avg_imag
        )

        fig_re, ax_re = plot_init(r"$\omega$ (cm$^{-1}$)", r"Re$[\varepsilon]$", "")
        ax_re.set_xlim(self.ωi, self.ωf)
        ax_re.plot(self.ω, re_curve, lw=1.0, c=SPECTRUM_COLOR, label="MLMD")
        ax_re.legend(frameon=False)

        fig_im, ax_im = plot_init(r"$\omega$ (cm$^{-1}$)", r"$-$Im$[\varepsilon]$", "")
        ax_im.set_xlim(self.ωi, self.ωf)
        ax_im.plot(self.ω, -im_curve, lw=1.0, c=SPECTRUM_COLOR, label="MLMD")
        ax_im.legend(frameon=False)

        if save_prefix is not None:
            save_prefix = Path(save_prefix)
            fig_re.savefig(save_prefix.with_name(save_prefix.name + "_epsilon_real.png"), bbox_inches="tight", dpi=300)
            fig_im.savefig(save_prefix.with_name(save_prefix.name + "_epsilon_imag.png"), bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig_re)
            plt.close(fig_im)

    def print_summary(self):
        eps_inf_par, eps_inf_perp = self._parallel_perpendicular(self.epsilon_inf)
        eps_ion_par, eps_ion_perp = self._parallel_perpendicular(self.epsilon_ion)
        eps_0_par, eps_0_perp = self._parallel_perpendicular(self.epsilon_0)
        print(f"Trajectory: {self.trajectory_path}")
        print(f"Frames: {self.n_t}")
        print(f"Timestep: {self.timestep_fs:.6g} fs")
        print(f"Temperature: {self.temp:.6g} K")
        print(f"Volume: {self.V:.6g} A^3")
        print(f"Backend: {self.backend}")
        print(f"Polarization key: {self.polarization_key}")
        print(f"Polarizability key: {self.polarizability_key}")
        print(f"BEC key: {self.bec_key}")
        print(f"Polar axis: {self.polar_axis_label}")
        print(f"Average epsilon_inf: {np.mean(self.epsilon_inf):.6g}")
        print(f"epsilon_inf parallel/perpendicular: {eps_inf_par:.6g} / {eps_inf_perp:.6g}")
        print(f"Average epsilon_ion: {np.mean(self.epsilon_ion):.6g}")
        print(f"epsilon_ion parallel/perpendicular: {eps_ion_par:.6g} / {eps_ion_perp:.6g}")
        print(f"Average epsilon_0: {np.mean(self.epsilon_0):.6g}")
        print(f"epsilon_0 parallel/perpendicular: {eps_0_par:.6g} / {eps_0_perp:.6g}")
        if self.do_IR and hasattr(self, "reference_ir_peaks_cm"):
            for key, value in self.reference_ir_peaks_cm.items():
                print(f"{key}: {value:.6g} cm^-1")


def plot_comparison(
    curves,
    output_prefix,
    freq_min,
    freq_max,
    do_smear,
    gaussian_broadening_cm,
    normalize_intensity=False,
    show=True,
    mode_resolved_payload=None,
    experimental_raman_curve=None,
    experimental_raman_peaks=None,
):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    include_mode_resolved = mode_resolved_payload is not None
    if include_mode_resolved:
        fig = plt.figure(figsize=(8.8, 9.6), constrained_layout=False)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95])
        ax_ir = fig.add_subplot(gs[0, 0])
        ax_raman = fig.add_subplot(gs[0, 1])
        ax_eps_re = fig.add_subplot(gs[1, 0], sharex=ax_ir)
        ax_eps_im = fig.add_subplot(gs[1, 1], sharex=ax_raman)
        ax_mode = fig.add_subplot(gs[2, :], sharex=ax_ir)
        axes = [ax_ir, ax_raman, ax_eps_re, ax_eps_im, ax_mode]
        plt.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.95, wspace=0.30, hspace=0.34)
    else:
        fig, axes_arr = plt.subplots(2, 2, figsize=(8.2, 7.2), sharex=True, constrained_layout=False)
        plt.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.92, wspace=0.30, hspace=0.34)
        ax_ir, ax_raman, ax_eps_re, ax_eps_im = axes_arr.ravel()
        ax_mode = None
        axes = [ax_ir, ax_raman, ax_eps_re, ax_eps_im]

    cmap = plt.get_cmap("tab10")
    all_ir = []
    all_raman = []
    all_re = []
    all_im = []

    for ax in axes:
        axis_settings(ax)
        ax.set_xlim(freq_min, freq_max)
        ax.grid(False)

    ax_ir.set_title("IR")
    ax_raman.set_title("Raman")
    ax_eps_re.set_title(r"Re $\varepsilon$")
    ax_eps_im.set_title(r"Imag $\varepsilon$")

    ax_ir.set_ylabel("Intensity (a.u.)")
    ax_raman.set_ylabel("Intensity (a.u.)")
    ax_eps_re.set_ylabel(r"$\varepsilon$ ($\varepsilon_0$)")
    ax_eps_im.set_ylabel(r"$-\mathrm{Im}\,\varepsilon$")
    ax_eps_re.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax_eps_im.set_xlabel(r"$\omega$ (cm$^{-1}$)")

    hide_y_ticks_and_values(ax_ir)
    hide_y_ticks_and_values(ax_raman)
    quartz_ref = load_allegro_pol_quartz_reference()

    for idx, curve in enumerate(curves):
        color = cmap(idx % 10)
        freq = curve.ω
        window_mask = (freq >= freq_min) & (freq <= freq_max)

        ir = gaussian_broaden(freq, curve.IR, gaussian_broadening_cm) if do_smear else curve.IR
        raman = gaussian_broaden(freq, curve.Raman, gaussian_broadening_cm) if do_smear else curve.Raman
        eps_re = (
            gaussian_broaden(freq, curve.epsilon_omega_avg_real, gaussian_broadening_cm)
            if do_smear
            else curve.epsilon_omega_avg_real
        )
        eps_im = (
            gaussian_broaden(freq, curve.epsilon_omega_avg_imag, gaussian_broadening_cm)
            if do_smear
            else curve.epsilon_omega_avg_imag
        )
        eps_im_plot = -eps_im

        # IR is conventionally shown as downward absorption in our comparison
        # plots, while Raman is shown as a positive intensity.
        ir = -np.abs(ir)
        raman = np.abs(raman)

        if normalize_intensity:
            ir_window = ir[window_mask] if np.any(window_mask) else ir
            raman_window = raman[window_mask] if np.any(window_mask) else raman
            ir_scale = np.nanmax(np.abs(ir_window))
            raman_scale = np.nanmax(np.abs(raman_window))
            if np.isfinite(ir_scale) and ir_scale > 0.0:
                ir = ir / ir_scale
            if np.isfinite(raman_scale) and raman_scale > 0.0:
                raman = raman / raman_scale

        all_ir.append(ir[window_mask] if np.any(window_mask) else ir)
        all_raman.append(raman[window_mask] if np.any(window_mask) else raman)
        all_re.append(eps_re[window_mask] if np.any(window_mask) else eps_re)
        all_im.append(eps_im_plot[window_mask] if np.any(window_mask) else eps_im_plot)

        ax_ir.plot(freq, ir, lw=1.8, color=color, label=curve.label)
        ax_ir.fill_between(freq, 0.0, ir, color=color, alpha=0.18)

        ax_raman.plot(freq, raman, lw=1.8, color=color, label=curve.label)
        ax_raman.fill_between(freq, 0.0, raman, color=color, alpha=0.18)

        ax_eps_re.plot(freq, eps_re, lw=1.8, color=color, label=curve.label)
        ax_eps_re.fill_between(freq, 0.0, eps_re, color=color, alpha=0.18)

        ax_eps_im.plot(freq, eps_im_plot, lw=1.8, color=color, label=curve.label)
        ax_eps_im.fill_between(freq, 0.0, eps_im_plot, color=color, alpha=0.18)

    if experimental_raman_curve is not None and all_raman:
        exp_freq, exp_intensity = experimental_raman_curve
        current_raman = np.concatenate(all_raman)
        current_finite = current_raman[np.isfinite(current_raman)]
        target = float(np.nanmax(current_finite)) if current_finite.size else 1.0
        if normalize_intensity:
            target = 1.0

        # Process the experimental Raman curve with the same frequency-grid and
        # Gaussian broadening convention used for the plotted comparison spectra.
        # This keeps the upper-right Raman comparison consistent with the
        # mode-resolved Raman subplot. If --no-smear is used, sigma is set to
        # zero so the helper only baseline-corrects, interpolates, and normalizes.
        exp_freq, exp_plot = prepare_experimental_mode_resolved_curve(
            exp_freq,
            exp_intensity,
            freq_min=freq_min,
            freq_max=freq_max,
            sigma_cm=gaussian_broadening_cm if do_smear else 0.0,
            target_max=target,
            n_grid=len(curves[0].ω) if curves else 4000,
        )
        ax_raman.plot(
            exp_freq,
            exp_plot,
            color=EXPERIMENTAL_RAMAN_COLOR,
            lw=2.0,
            ls=":",
            label="Exp. Raman",
            zorder=60,
        )
        all_raman.append(exp_plot)

    if quartz_ref is not None:
        ref_ir_sticks = np.asarray(quartz_ref["ir_intensity"], dtype=np.float64)
        ref_ir = np.zeros_like(curves[0].ω, dtype=np.float64)
        for f_mode, intensity in zip(quartz_ref["ir_freq_cm"], ref_ir_sticks):
            ref_ir += gaussian(curves[0].ω, abs(float(intensity)), float(f_mode), gaussian_broadening_cm)
        ref_ir = -np.abs(ref_ir)
        if normalize_intensity:
            ref_window = (curves[0].ω >= freq_min) & (curves[0].ω <= freq_max)
            ref_scale = np.nanmax(np.abs(ref_ir[ref_window])) if np.any(ref_window) else np.nanmax(np.abs(ref_ir))
            if np.isfinite(ref_scale) and ref_scale > 0.0:
                ref_ir = ref_ir / ref_scale
        ax_ir.plot(
            curves[0].ω,
            ref_ir,
            lw=1.5,
            ls="--",
            color=REFERENCE_COLOR,
            label=quartz_ref["label"],
            zorder=50,
        )
        ref_re = np.asarray(quartz_ref["eps_re"], dtype=np.float64)
        ref_im = np.asarray(quartz_ref["eps_im"], dtype=np.float64)
        ax_eps_re.plot(
            quartz_ref["eps_re_freq_cm"],
            ref_re,
            lw=1.5,
            ls="--",
            color=REFERENCE_COLOR,
            label=quartz_ref["label"],
            zorder=50,
        )
        ax_eps_im.plot(
            quartz_ref["eps_im_freq_cm"],
            ref_im,
            lw=1.5,
            ls="--",
            color=REFERENCE_COLOR,
            label=quartz_ref["label"],
            zorder=50,
        )
        all_ir.append(ref_ir[(curves[0].ω >= freq_min) & (curves[0].ω <= freq_max)])
        all_re.append(ref_re[(quartz_ref["eps_re_freq_cm"] >= freq_min) & (quartz_ref["eps_re_freq_cm"] <= freq_max)])
        all_im.append(ref_im[(quartz_ref["eps_im_freq_cm"] >= freq_min) & (quartz_ref["eps_im_freq_cm"] <= freq_max)])

    ax_ir.legend(frameon=False, loc="upper left", fontsize=10)
    ax_raman.legend(frameon=False, loc="upper right", fontsize=10)
    ax_eps_re.legend(frameon=False, loc="upper left", fontsize=10)
    ax_eps_im.legend(frameon=False, loc="upper left", fontsize=10)

    if all_ir:
        ir_concat = np.concatenate(all_ir)
        raman_concat = np.concatenate(all_raman)
        re_concat = np.concatenate(all_re)
        im_concat = np.concatenate(all_im)
        ir_finite = ir_concat[np.isfinite(ir_concat)]
        ir_min = float(np.nanmin(ir_finite)) if ir_finite.size else -1.0
        ax_ir.set_ylim(ir_min * 1.04 if ir_min < 0.0 else -1.0, 0.0)
        ax_raman.set_ylim(*positive_limits(raman_concat[np.isfinite(raman_concat)], pad_frac=0.04))
        ax_eps_re.set_ylim(*ylim_range(re_concat[np.isfinite(re_concat)]))
        ax_eps_im.set_ylim(*positive_limits(im_concat[np.isfinite(im_concat)]))

        hide_y_ticks_and_values(ax_ir)
        hide_y_ticks_and_values(ax_raman)

    if ax_mode is not None:
        plot_mode_resolved_subplot(
            ax_mode,
            mode_resolved_payload,
            curves,
            freq_min=freq_min,
            freq_max=freq_max,
            sigma_cm=gaussian_broadening_cm,
            experimental_curve=experimental_raman_curve,
            experimental_peaks=experimental_raman_peaks,
        )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return png_path, pdf_path


def write_comparison_summary(curves, output_prefix):
    csv_path = Path(output_prefix).with_name(Path(output_prefix).name + "_summary.csv")
    fieldnames = [
        "label",
        "trajectory",
        "n_frames",
        "temperature_K",
        "timestep_fs",
        "polar_axis",
        "epsilon_inf_avg",
        "epsilon_inf_parallel",
        "epsilon_inf_perpendicular",
        "epsilon_ion_avg",
        "epsilon_ion_parallel",
        "epsilon_ion_perpendicular",
        "epsilon_0_avg",
        "epsilon_0_parallel",
        "epsilon_0_perpendicular",
        "reference_peak_source",
        "omega_1_cm-1",
        "omega_2_cm-1",
        "omega_3_cm-1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            eps_inf_par, eps_inf_perp = curve._parallel_perpendicular(curve.epsilon_inf)
            eps_ion_par, eps_ion_perp = curve._parallel_perpendicular(curve.epsilon_ion)
            eps_0_par, eps_0_perp = curve._parallel_perpendicular(curve.epsilon_0)
            writer.writerow(
                {
                    "label": curve.label,
                    "trajectory": str(curve.trajectory_path),
                    "n_frames": curve.n_t,
                    "temperature_K": curve.temp,
                    "timestep_fs": curve.timestep_fs,
                    "polar_axis": curve.polar_axis_label,
                    "epsilon_inf_avg": float(np.mean(curve.epsilon_inf)),
                    "epsilon_inf_parallel": eps_inf_par,
                    "epsilon_inf_perpendicular": eps_inf_perp,
                    "epsilon_ion_avg": float(np.mean(curve.epsilon_ion)),
                    "epsilon_ion_parallel": eps_ion_par,
                    "epsilon_ion_perpendicular": eps_ion_perp,
                    "epsilon_0_avg": float(np.mean(curve.epsilon_0)),
                    "epsilon_0_parallel": eps_0_par,
                    "epsilon_0_perpendicular": eps_0_perp,
                    "reference_peak_source": "broadened_IR_magnitude" if curve.do_IR else "",
                    "omega_1_cm-1": curve.reference_ir_peaks_cm.get("omega_1_cm-1", np.nan) if curve.do_IR else np.nan,
                    "omega_2_cm-1": curve.reference_ir_peaks_cm.get("omega_2_cm-1", np.nan) if curve.do_IR else np.nan,
                    "omega_3_cm-1": curve.reference_ir_peaks_cm.get("omega_3_cm-1", np.nan) if curve.do_IR else np.nan,
                }
            )
    return csv_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        help="Comparison mode: provide a label and annotated extxyz path. Repeat for multiple curves.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Annotated extxyz trajectory to analyse. Defaults to the newest SiO2 production.annotated.extxyz.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for saved .npz/.json/.csv/(optional .png) outputs. Defaults next to the input trajectory.",
    )
    parser.add_argument("--temperature-k", type=float, default=None, help="Override trajectory temperature in K.")
    parser.add_argument("--timestep-fs", type=float, default=None, help="Override sampling timestep in fs.")
    parser.add_argument("--freq-min", type=float, default=0.0, help="Minimum plotted frequency in cm^-1.")
    parser.add_argument("--freq-max", type=float, default=1400.0, help="Maximum plotted frequency in cm^-1.")
    parser.add_argument("--autocorr-ensembles", type=int, default=1, help="Number of trajectory prefixes to average.")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--gaussian-broadening-cm", type=float, default=DEFAULT_GAUSSIAN_BROADENING_CM)
    parser.add_argument(
        "--polar-axis",
        default="z",
        help="Polar axis for parallel/perpendicular dielectric summaries: x/y/z or a/b/c. Default: z.",
    )
    parser.add_argument("--no-ir", action="store_true", help="Skip IR / dielectric analysis.")
    parser.add_argument("--no-raman", action="store_true", help="Skip Raman analysis.")
    parser.add_argument("--no-smear", action="store_true", help="Disable Gaussian broadening in plots.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows.")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots alongside the numerical outputs.")
    parser.add_argument(
        "--normalize-intensity",
        action="store_true",
        help="In comparison mode, normalize each IR/Raman curve independently to unit peak height.",
    )
    parser.add_argument(
        "--mode-resolved-dir",
        type=Path,
        default=None,
        help="Directory containing mode-resolved Raman CSV outputs. Adds a third-row mode-resolved Raman subplot in comparison mode.",
    )
    parser.add_argument(
        "--experimental-raman-source",
        default=DEFAULT_EXPERIMENTAL_RAMAN_URL,
        help="URL or local path for a two-column experimental Raman curve. Defaults to the ENS-Lyon quartz text spectrum.",
    )
    parser.add_argument(
        "--no-experimental-raman-curve",
        action="store_true",
        help="Disable plotting the full experimental Raman curve.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    show = not args.no_show
    smear = not args.no_smear

    if args.curve:
        curves = []
        for label, path_str in args.curve:
            trajectory_path = Path(path_str).expanduser().resolve()
            curve_output_prefix = None
            if args.output_prefix is not None:
                base = args.output_prefix.resolve()
                curve_output_prefix = base.with_name(f"{base.name}_{label}")
            spec = Spectroscopy(
                trajectory_path=trajectory_path,
                temp_k=args.temperature_k,
                freq_min_cm=args.freq_min,
                freq_max_cm=args.freq_max,
                do_ir=not args.no_ir,
                do_raman=not args.no_raman,
                timestep_fs=args.timestep_fs,
                autocorr_ensembles=args.autocorr_ensembles,
                backend=args.backend,
                save_prefix=curve_output_prefix,
                gaussian_broadening_cm=args.gaussian_broadening_cm,
                polar_axis=args.polar_axis,
            )
            spec.label = label
            curves.append(spec)
            spec.print_summary()

        if args.output_prefix is None:
            output_prefix = Path.cwd() / "sio2_spectroscopy_compare"
        else:
            output_prefix = args.output_prefix.resolve()

        mode_resolved_dir = infer_mode_resolved_dir(args.mode_resolved_dir)
        mode_resolved_payload = load_mode_resolved_results(mode_resolved_dir)
        experimental_raman_curve = None
        if not args.no_experimental_raman_curve:
            experimental_raman_curve = load_experimental_raman_curve(
                args.experimental_raman_source,
                cache_dir=Path(args.output_prefix).parent if args.output_prefix is not None else Path.cwd(),
            )
            if experimental_raman_curve is None:
                print(f"Warning: could not load experimental Raman curve from {args.experimental_raman_source!r}")
        png_path, pdf_path = plot_comparison(
            curves,
            output_prefix=output_prefix,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            do_smear=smear,
            gaussian_broadening_cm=args.gaussian_broadening_cm,
            normalize_intensity=args.normalize_intensity,
            show=show,
            mode_resolved_payload=mode_resolved_payload,
            experimental_raman_curve=experimental_raman_curve,
            experimental_raman_peaks=load_experimental_raman_peaks(),
        )
        summary_csv = write_comparison_summary(curves, output_prefix)
        print(f"Saved comparison plot: {png_path}")
        print(f"Saved comparison plot: {pdf_path}")
        print(f"Saved comparison summary: {summary_csv}")
        return

    trajectory_path = args.input.resolve() if args.input is not None else resolve_default_input()
    if args.output_prefix is None:
        output_prefix = trajectory_path.with_suffix("")
    else:
        output_prefix = args.output_prefix.resolve()

    spec = Spectroscopy(
        trajectory_path=trajectory_path,
        temp_k=args.temperature_k,
        freq_min_cm=args.freq_min,
        freq_max_cm=args.freq_max,
        do_ir=not args.no_ir,
        do_raman=not args.no_raman,
        timestep_fs=args.timestep_fs,
        autocorr_ensembles=args.autocorr_ensembles,
        backend=args.backend,
        save_prefix=output_prefix,
        gaussian_broadening_cm=args.gaussian_broadening_cm,
        polar_axis=args.polar_axis,
    )

    spec.print_summary()

    if spec.do_IR:
        spec.plot_autocorr_P(
            show=show,
            save_path=output_prefix.with_name(output_prefix.name + "_P_autocorr.png") if args.save_plots else None,
        )
        spec.plot_IR(
            do_smear=smear,
            show=show,
            save_path=output_prefix.with_name(output_prefix.name + "_IR.png") if args.save_plots else None,
        )
        spec.plot_epsilon(do_smear=smear, show=show, save_prefix=output_prefix if args.save_plots else None)
    if spec.do_Raman:
        spec.plot_autocorr_alpha(
            show=show,
            save_path=output_prefix.with_name(output_prefix.name + "_alpha_autocorr.png") if args.save_plots else None,
        )
        spec.plot_Raman(
            do_smear=smear,
            show=show,
            save_path=output_prefix.with_name(output_prefix.name + "_Raman.png") if args.save_plots else None,
        )


if __name__ == "__main__":
    main()
