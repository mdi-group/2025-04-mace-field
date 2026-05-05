#!/usr/bin/env python3
"""Compare hysteresis loops from annotated BaTiO3 trajectories.

This script reads one or more `hysteresis.annotated.extxyz` trajectories,
extracts the chosen electric-field and polarization components, overlays the
loops, and writes out a CSV summary with coercive fields and remanent
polarization estimates for each curve.

Example
-------
python plot_hysteresis.py \
  --curve omat /path/to/omat/hysteresis.annotated.extxyz \
  --curve direct /path/to/direct/hysteresis.annotated.extxyz \
  --curve finetuned /path/to/finetuned/hysteresis.annotated.extxyz \
  --skip 50 \
  --output-prefix plots/batio3_hysteresis_compare
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


POL_TO_UC_CM2 = 1602.176634
EFIELD_TO_MV_CM = 100.0


def parse_axis(axis: str) -> int:
    lookup = {"x": 0, "y": 1, "z": 2, "a": 0, "b": 1, "c": 2}
    try:
        return lookup[axis.lower()]
    except KeyError as exc:
        raise ValueError(f"Axis must be one of x, y, z, a, b, c; got {axis!r}") from exc


def resolve_reference_root() -> Path:
    return Path(__file__).resolve().parent / "allegro-pol-reference"


def _extract_thermo_rows_from_lammps_dat(path: Path):
    rows = []
    in_table = False
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Step" and "v_efield" in parts and "v_Pz" in parts:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("Loop time"):
            break
        try:
            [float(value) for value in parts]
        except ValueError:
            continue
        if len(parts) >= 10:
            rows.append(parts[:10])
    return rows


def load_allegro_pol_mlmd_reference():
    folder = resolve_reference_root() / "BaTiO3-8640-T300"
    files = sorted(
        folder.glob("BaTiO3-*.dat"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )
    if not files:
        return None

    efield_series = []
    pol_series = []
    for path in files:
        rows = _extract_thermo_rows_from_lammps_dat(path)
        if not rows:
            continue
        efield = np.asarray([float(row[6]) for row in rows], dtype=float) * EFIELD_TO_MV_CM
        pol = np.asarray([float(row[7]) for row in rows], dtype=float) * POL_TO_UC_CM2
        efield_series.append(efield)
        pol_series.append(pol)

    if not efield_series:
        return None

    n = min(len(x) for x in efield_series)
    efield_stack = np.vstack([x[:n] for x in efield_series])
    pol_stack = np.vstack([x[:n] for x in pol_series])
    return {
        "label": "Allegro-pol",
        "path": str(folder),
        "E": np.mean(efield_stack, axis=0),
        "P": np.mean(pol_stack, axis=0),
    }


def load_digitized_dft_reference():
    root = resolve_reference_root()
    preferred = [
        root / "allegro_pol_bto_fig3a_dft_digitized_dense.csv",
        root / "allegro_pol_bto_fig3a_dft_digitized.csv",
    ]
    path = next((candidate for candidate in preferred if candidate.exists()), None)
    if path is None:
        return None
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return None
    if rows.ndim == 0:
        rows = np.array([rows], dtype=rows.dtype)
    names = rows.dtype.names or ()

    def pick(*candidates):
        for candidate in candidates:
            if candidate in names:
                return candidate
        raise ValueError(
            f"Digitized DFT CSV at {path} is missing expected columns. "
            f"Found columns: {names}"
        )

    efield_key = pick("efield_MV_cm", "Ez_MV_per_cm", "E_MV_cm")
    pol_key = pick("polarization_uC_cm2", "Pz_uC_per_cm2", "P_uC_cm2")
    branch_key = pick("branch")

    efield = np.asarray(rows[efield_key], dtype=float)
    pol = np.asarray(rows[pol_key], dtype=float)
    branch = np.asarray(rows[branch_key], dtype=str)
    return {"label": "DFT", "path": path, "E": efield, "P": pol, "branch": branch}


def median_curve(ex, px, n_bins):
    ex = np.asarray(ex, dtype=float)
    px = np.asarray(px, dtype=float)
    if ex.size < 3:
        return np.array([]), np.array([])

    edges = np.linspace(ex.min(), ex.max(), n_bins + 1)
    idx = np.digitize(ex, edges) - 1
    em, pm = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 3:
            em.append(np.median(ex[mask]))
            pm.append(np.median(px[mask]))
    if not em:
        return np.array([]), np.array([])
    order = np.argsort(em)
    return np.asarray(em)[order], np.asarray(pm)[order]


def crossings_x_at_y0(x, y):
    xs = []
    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == 0:
            xs.append(x[i])
        elif y1 * y2 < 0:
            t = abs(y1) / (abs(y1) + abs(y2))
            xs.append(x[i] * (1 - t) + x[i + 1] * t)
    return np.asarray(xs, dtype=float)


def values_y_at_x0(x, y):
    ys = []
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        if x1 == 0:
            ys.append(y[i])
        elif x1 * x2 < 0:
            t = abs(x1) / (abs(x1) + abs(x2))
            ys.append(y[i] * (1 - t) + y[i + 1] * t)
    return np.asarray(ys, dtype=float)


def closest_to_zero(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.argmin(np.abs(values))]) if values.size else np.nan


def robust_limits(values, pad_frac=0.06):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    span = max(hi - lo, 1e-9)
    return lo - pad_frac * span, hi + pad_frac * span


def choose_marker_center(x, y, target_x=None, prefer_vertical=False, forward=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 6:
        return None
    if target_x is None:
        center = int(round(0.68 * (x.size - 1))) if forward else int(round(0.32 * (x.size - 1)))
        return max(1, min(center, x.size - 2))

    window = max(np.ptp(x) * 0.08, 1.0)
    candidates = np.where(np.abs(x - target_x) <= window)[0]
    if candidates.size == 0:
        candidates = np.array([int(np.argmin(np.abs(x - target_x)))])
    candidates = candidates[(candidates > 0) & (candidates < x.size - 1)]
    if candidates.size == 0:
        return max(1, min(int(np.argmin(np.abs(x - target_x))), x.size - 2))

    if prefer_vertical:
        left = np.maximum(candidates - 1, 0)
        right = np.minimum(candidates + 1, x.size - 1)
        dx = np.abs(x[right] - x[left])
        dy = np.abs(y[right] - y[left])
        score = dy / np.maximum(dx, 1e-9)
        center = int(candidates[np.argmax(score)])
    else:
        center = int(candidates[np.argmin(np.abs(x[candidates] - target_x))])
    return max(1, min(center, x.size - 2))


def add_direction_arrow(ax, x, y, color, forward=True, target_x=None, prefer_vertical=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    center = choose_marker_center(x, y, target_x=target_x, prefer_vertical=prefer_vertical, forward=forward)
    if center is None:
        return
    if prefer_vertical:
        if forward:
            start = center
            end = min(x.size - 1, center + 1)
        else:
            start = center
            end = max(0, center - 1)
        x_plot = 0.5 * (x[start] + x[end])
        y_plot = 0.5 * (y[start] + y[end])
        dx = x[end] - x[start]
        dy = y[end] - y[start]
    else:
        delta = max(1, x.size // 28)
        if forward:
            start = max(0, center - delta)
            end = min(x.size - 1, center + delta)
        else:
            start = min(x.size - 1, center + delta)
            end = max(0, center - delta)
        if start == end:
            return
        x_plot = x[end]
        y_plot = y[end]
        dx = x[end] - x[start]
        dy = y[end] - y[start]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    marker_angle = angle_deg - 90.0
    ax.scatter(
        [x_plot],
        [y_plot],
        s=170,
        marker=(3, 0, marker_angle),
        facecolor=color,
        edgecolor=color,
        linewidths=0.0,
        zorder=60,
    )


def load_curve(path: Path, field_axis: int, pol_axis: int, skip: int):
    frames = read(str(path), ":")
    if not frames:
        raise ValueError(f"No frames found in {path}")
    if skip >= len(frames):
        raise ValueError(f"Skip={skip} removes all frames from {path}")

    frames = frames[skip:]
    efield = np.asarray([frame.info["MACE_electric_field"][field_axis] for frame in frames], dtype=float)
    pol = np.asarray([frame.info["MACE_polarization"][pol_axis] for frame in frames], dtype=float)

    efield *= EFIELD_TO_MV_CM
    pol *= -POL_TO_UC_CM2
    return frames, efield, pol


def analyze_curve(label: str, path: Path, efield, pol, n_bins: int):
    dE = np.gradient(efield)
    up_mask = dE > 0
    down_mask = dE < 0

    eu, pu = efield[up_mask], pol[up_mask]
    ed, pd = efield[down_mask], pol[down_mask]

    eu_m, pu_m = median_curve(eu, pu, n_bins)
    ed_m, pd_m = median_curve(ed, pd, n_bins)

    ec_up = closest_to_zero(crossings_x_at_y0(eu_m, pu_m))
    ec_down = closest_to_zero(crossings_x_at_y0(ed_m, pd_m))
    pr_up = closest_to_zero(values_y_at_x0(eu_m, pu_m))
    pr_down = closest_to_zero(values_y_at_x0(ed_m, pd_m))

    return {
        "label": label,
        "path": str(path),
        "n_frames": int(len(efield)),
        "E": efield,
        "P": pol,
        "Eu": eu,
        "Pu": pu,
        "Ed": ed,
        "Pd": pd,
        "Eu_m": eu_m,
        "Pu_m": pu_m,
        "Ed_m": ed_m,
        "Pd_m": pd_m,
        "Ec_up_MV_cm": ec_up,
        "Ec_down_MV_cm": ec_down,
        "Ec_abs_avg_MV_cm": float(np.nanmean(np.abs([ec_up, ec_down]))),
        "Pr_up_uC_cm2": pr_up,
        "Pr_down_uC_cm2": pr_down,
        "Pr_abs_avg_uC_cm2": float(np.nanmean(np.abs([pr_up, pr_down]))),
    }


def plot_curves(curves, output_prefix: Path, title: str, show_raw: bool, raw_alpha: float):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90)

    cmap = plt.get_cmap("tab10")
    all_e = []
    all_p = []
    dft_ref = load_digitized_dft_reference()

    for idx, curve in enumerate(curves):
        color = cmap(idx % 10)
        all_e.append(curve["E"])
        all_p.append(curve["P"])

        ax.plot(curve["Eu_m"], curve["Pu_m"], color=color, lw=2.2, label=curve["label"])
        ax.plot(curve["Ed_m"], curve["Pd_m"], color=color, lw=2.2)
        ec_half = 0.5 * curve["Ec_abs_avg_MV_cm"]
        add_direction_arrow(ax, curve["Ed_m"], curve["Pd_m"], color=color, forward=False, target_x=ec_half)
        add_direction_arrow(
            ax,
            curve["Ed_m"],
            curve["Pd_m"],
            color=color,
            forward=False,
            target_x=curve["Ec_down_MV_cm"],
            prefer_vertical=True,
        )
        add_direction_arrow(ax, curve["Eu_m"], curve["Pu_m"], color=color, forward=True, target_x=-ec_half)
        add_direction_arrow(
            ax,
            curve["Eu_m"],
            curve["Pu_m"],
            color=color,
            forward=True,
            target_x=curve["Ec_up_MV_cm"],
            prefer_vertical=True,
        )

    if dft_ref is not None:
        upper_mask = np.char.lower(dft_ref["branch"]) == "upper"
        lower_mask = np.char.lower(dft_ref["branch"]) == "lower"
        upper_idx = np.where(upper_mask)[0]
        lower_idx = np.where(lower_mask)[0]
        if upper_idx.size:
            upper_zero = upper_idx[np.argmin(np.abs(dft_ref["E"][upper_idx]))]
        else:
            upper_zero = None
        if lower_idx.size:
            lower_zero = lower_idx[np.argmin(np.abs(dft_ref["E"][lower_idx]))]
        else:
            lower_zero = None

        dft_indices = [idx for idx in (upper_zero, lower_zero) if idx is not None]
        dft_e = dft_ref["E"][dft_indices]
        dft_p = dft_ref["P"][dft_indices]
        ax.plot(
            dft_e,
            dft_p,
            color="0.15",
            lw=0.0,
            ls="None",
            marker="o",
            ms=7.0,
            label=dft_ref["label"],
            zorder=80,
        )
        all_e.append(dft_e)
        all_p.append(dft_p)

    all_e = np.concatenate(all_e) if all_e else np.array([0.0])
    all_p = np.concatenate(all_p) if all_p else np.array([0.0])

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, ls=":", lw=0.6, color="0.85")
    ax.axvline(0, color="0.35", lw=1.0)
    ax.axhline(0, color="0.35", lw=1.0)
    ax.set_xlabel(r"Electric field, $E$ (MV/cm)", fontsize=12)
    ax.set_ylabel(r"Polarization, $P$ ($\mu$C/cm$^2$)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(*robust_limits(all_e))
    ax.set_ylim(*robust_limits(all_p))
    ax.legend(frameon=False, fontsize=11)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    return png_path, pdf_path


def write_metrics_csv(curves, output_prefix: Path):
    csv_path = output_prefix.with_name(output_prefix.name + "_metrics.csv")
    fieldnames = [
        "label",
        "path",
        "n_frames",
        "Ec_up_MV_cm",
        "Ec_down_MV_cm",
        "Ec_abs_avg_MV_cm",
        "Pr_up_uC_cm2",
        "Pr_down_uC_cm2",
        "Pr_abs_avg_uC_cm2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            writer.writerow({key: curve[key] for key in fieldnames})
    return csv_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="Curve to overlay: provide a legend label and an annotated extxyz path. Repeat for multiple datasets.",
    )
    parser.add_argument("--skip", type=int, default=0, help="Discard the first N frames from every trajectory.")
    parser.add_argument("--field-axis", default="z", help="Electric-field component to plot: x/y/z or a/b/c.")
    parser.add_argument("--polarization-axis", default="z", help="Polarization component to plot: x/y/z or a/b/c.")
    parser.add_argument("--n-bins", type=int, default=80, help="Number of E bins for median sweep curves.")
    parser.add_argument("--title", default="BaTiO3 polarization-field hysteresis", help="Plot title.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/brad/repositories/2025-04-mace-field/scripts/BaTiO3/plots/batio3_hysteresis_compare"),
        help="Prefix for .png/.pdf plot outputs and the metrics CSV.",
    )
    parser.add_argument("--no-raw", action="store_true", help="Hide the faint raw scatter points.")
    parser.add_argument("--raw-alpha", type=float, default=0.15, help="Alpha for raw scatter points if shown.")
    return parser


def main():
    args = build_parser().parse_args()
    field_axis = parse_axis(args.field_axis)
    pol_axis = parse_axis(args.polarization_axis)
    output_prefix = args.output_prefix.expanduser().resolve()

    curves = []
    for label, path_str in args.curve:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Trajectory not found: {path}")
        _, efield, pol = load_curve(path, field_axis=field_axis, pol_axis=pol_axis, skip=args.skip)
        curves.append(analyze_curve(label, path, efield, pol, n_bins=args.n_bins))

    png_path, pdf_path = plot_curves(
        curves,
        output_prefix=output_prefix,
        title=args.title,
        show_raw=not args.no_raw,
        raw_alpha=args.raw_alpha,
    )
    csv_path = write_metrics_csv(curves, output_prefix=output_prefix)

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
