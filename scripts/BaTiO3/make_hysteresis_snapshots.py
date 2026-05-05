#!/usr/bin/env python3
"""Create representative BaTiO3 hysteresis snapshots and switching diagnostics.

This script is intended for Supplementary Information figures accompanying the
finite-field BaTiO3 hysteresis trajectories. It selects representative frames
along the loop, writes those structures to disk, and produces:

1. a snapshot montage of key structures along the loop
2. a time-series diagnostic figure showing the global polarization and a simple
   local switching proxy based on average Ti off-centering along the polar axis
3. a CSV manifest summarizing the chosen frames

Example
-------
python make_hysteresis_snapshots.py \
  --input ../LAMMPs/MD/runs/BaTiO3-.../BaTiO3-mp-5986/hysteresis.annotated.extxyz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from ase.visualize.plot import plot_atoms


POL_TO_UC_CM2 = 1602.176634
EFIELD_TO_MV_CM = 100.0


def resolve_default_input() -> Path:
    runs_root = Path(__file__).resolve().parents[1] / "LAMMPs" / "MD" / "runs"
    candidates = sorted(
        runs_root.glob("BaTiO3-*/BaTiO3-mp-5986/hysteresis.annotated.extxyz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not find any BaTiO3 hysteresis.annotated.extxyz trajectories.")
    return candidates[0]


def load_trajectory(path: Path):
    frames = read(str(path), ":")
    if not frames:
        raise ValueError(f"No frames found in {path}")
    time_ps = np.asarray([float(at.info.get("time_ps", i)) for i, at in enumerate(frames)], dtype=float)
    efield = np.asarray([at.info["MACE_electric_field"][2] for at in frames], dtype=float) * EFIELD_TO_MV_CM
    pol = -np.asarray([at.info["MACE_polarization"][2] for at in frames], dtype=float) * POL_TO_UC_CM2
    temp = np.asarray([float(at.info.get("temperature_K", np.nan)) for at in frames], dtype=float)
    return frames, time_ps, efield, pol, temp


def nearest_image_vector(cell, origin, target):
    frac = np.linalg.solve(cell.T, (target - origin).T).T
    frac -= np.round(frac)
    return frac @ cell


def build_ti_o_mapping(frame):
    symbols = np.asarray(frame.symbols)
    ti_indices = np.where(symbols == "Ti")[0]
    o_indices = np.where(symbols == "O")[0]
    positions = frame.get_positions()
    cell = np.asarray(frame.cell)
    mapping = []
    for ti_idx in ti_indices:
        rel = np.asarray(
            [nearest_image_vector(cell, positions[ti_idx], positions[o_idx]) for o_idx in o_indices],
            dtype=float,
        )
        dist = np.linalg.norm(rel, axis=1)
        nearest = np.argsort(dist)[:6]
        mapping.append((ti_idx, o_indices[nearest]))
    return mapping


def compute_ti_offcentering(frames, mapping, axis=2):
    signed = np.empty(len(frames), dtype=float)
    magnitude = np.empty(len(frames), dtype=float)
    for frame_idx, frame in enumerate(frames):
        positions = frame.get_positions()
        cell = np.asarray(frame.cell)
        local_vectors = []
        for ti_idx, o_group in mapping:
            rel = np.asarray(
                [nearest_image_vector(cell, positions[ti_idx], positions[o_idx]) for o_idx in o_group],
                dtype=float,
            )
            offcenter = -rel.mean(axis=0)
            local_vectors.append(offcenter)
        local_vectors = np.asarray(local_vectors, dtype=float)
        signed[frame_idx] = np.mean(local_vectors[:, axis])
        magnitude[frame_idx] = np.mean(np.linalg.norm(local_vectors, axis=1))
    return signed, magnitude


def best_index(mask, score):
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        return None
    local = np.argmin(score[candidates])
    return int(candidates[local])


def contiguous_runs(indices):
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return [chunk for chunk in np.split(indices, splits) if chunk.size]


def nearest_time_index(time_ps, target_ps):
    return int(np.argmin(np.abs(np.asarray(time_ps, dtype=float) - float(target_ps))))


def choose_representative_frames(time_ps, efield, pol):
    dE = np.gradient(efield)
    ascending = dE > 0

    negative_field_idx = nearest_time_index(time_ps, 100.0)
    zero_field_idx = nearest_time_index(time_ps, 150.0)
    positive_field_idx = nearest_time_index(time_ps, 200.0)

    positive_half_cycle = ascending & (np.asarray(time_ps) >= time_ps[zero_field_idx] - 1e-9)
    if not np.any(positive_half_cycle):
        raise ValueError("Could not identify the post-150 ps ascending half-cycle for the positive-field switch.")

    dpol_dt = np.abs(np.gradient(pol, time_ps))
    switch_candidates = np.where(positive_half_cycle & (efield > 0.0))[0]
    if switch_candidates.size == 0:
        raise ValueError("Could not identify the positive-field switching event.")
    switch_peak = int(switch_candidates[np.argmax(dpol_dt[switch_candidates])])

    strong_switch = np.where(positive_half_cycle & (dpol_dt >= 0.4 * dpol_dt[switch_peak]))[0]
    runs = contiguous_runs(strong_switch)
    switch_run = next((run for run in runs if run[0] <= switch_peak <= run[-1]), np.array([switch_peak], dtype=int))

    before_switch = np.where((np.arange(len(time_ps)) >= zero_field_idx) & (np.arange(len(time_ps)) < switch_run[0]))[0]
    pre_idx = int(before_switch[-1]) if before_switch.size else int(max(zero_field_idx, switch_run[0] - 1))

    mid_pos = int(round(0.5 * (switch_run.size - 1)))
    mid_idx = int(switch_run[mid_pos])
    end_idx = int(switch_run[-1])

    selections = [
        ("negative_field", negative_field_idx),
        ("zero_field", zero_field_idx),
        ("pre_switch", pre_idx),
        ("switch_mid", mid_idx),
        ("switch_end", end_idx),
        ("positive_field", positive_field_idx),
    ]

    deduped = []
    used = set()
    for label, idx in selections:
        if idx in used:
            continue
        used.add(idx)
        deduped.append((label, idx))
    return deduped


def write_snapshot_structures(frames, snapshot_rows, output_dir: Path):
    snapshot_dir = output_dir / "structures"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for old_file in snapshot_dir.glob("*.xyz"):
        old_file.unlink()
    for row in snapshot_rows:
        at = frames[row["frame_index"]].copy()
        filename = snapshot_dir / f"{row['order']:02d}_{row['label']}.xyz"
        write(filename, at)
        row["structure_path"] = str(filename)


def build_snapshot_rows(labels_and_indices, time_ps, efield, pol, temp, ti_offset_z, ti_offset_mag):
    rows = []
    for order, (label, idx) in enumerate(labels_and_indices, start=1):
        rows.append(
            {
                "order": order,
                "label": label,
                "frame_index": int(idx),
                "time_ps": float(time_ps[idx]),
                "electric_field_MV_cm": float(efield[idx]),
                "polarization_uC_cm2": float(pol[idx]),
                "temperature_K": float(temp[idx]),
                "avg_ti_offcenter_z_A": float(ti_offset_z[idx]),
                "avg_ti_offcenter_mag_A": float(ti_offset_mag[idx]),
            }
        )
    return rows


def save_snapshot_manifest(rows, output_dir: Path):
    csv_path = output_dir / "batio3_snapshot_summary.csv"
    fieldnames = [
        "order",
        "label",
        "frame_index",
        "time_ps",
        "electric_field_MV_cm",
        "polarization_uC_cm2",
        "temperature_K",
        "avg_ti_offcenter_z_A",
        "avg_ti_offcenter_mag_A",
        "structure_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def make_snapshot_panel(frames, rows, output_dir: Path, rotation: str):
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        at = frames[row["frame_index"]]
        plot_atoms(at, ax, rotation=rotation, radii=0.42, show_unit_cell=2)
        ax.set_axis_off()
        ax.set_title(
            (
                f"{row['label'].replace('_', ' ').title()}\n"
                f"t = {row['time_ps']:.2f} ps, "
                f"E = {row['electric_field_MV_cm']:.2f} MV/cm, "
                f"P = {row['polarization_uC_cm2']:.2f} $\\mu$C/cm$^2$"
            ),
            fontsize=10,
        )
    for ax in axes[len(rows) :]:
        ax.set_axis_off()
    fig.tight_layout()
    out_path = output_dir / "batio3_hysteresis_snapshots.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_switching_trace(time_ps, efield, pol, ti_offset_z, rows, output_dir: Path):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.90)

    def normalize(series):
        scale = float(np.nanmax(np.abs(series)))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return np.asarray(series, dtype=float) / scale, scale

    pol_norm, pol_scale = normalize(pol)
    efield_norm, efield_scale = normalize(efield)
    ti_norm, ti_scale = normalize(ti_offset_z)

    style = [
        (pol_norm, "tab:blue", rf"$P_z$ / {pol_scale:.1f} $\mu$C cm$^{{-2}}$"),
        (efield_norm, "tab:orange", rf"$E_z$ / {efield_scale:.2f} MV cm$^{{-1}}$"),
        (ti_norm, "tab:green", rf"$\langle \delta z_{{\rm Ti}} \rangle$ / {ti_scale:.3f} $\AA$"),
    ]

    for values, color, label in style:
        ax.plot(time_ps, values, color=color, lw=2.4, label=label)

    snapshot_lookup = {
        "tab:blue": (pol_norm, "polarization_uC_cm2", pol_scale),
        "tab:orange": (efield_norm, "electric_field_MV_cm", efield_scale),
        "tab:green": (ti_norm, "avg_ti_offcenter_z_A", ti_scale),
    }
    for color, (_, row_key, scale) in snapshot_lookup.items():
        ax.scatter(
            [row["time_ps"] for row in rows],
            [row[row_key] / scale for row in rows],
            color=color,
            s=30,
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.axvline(150.0, color="0.55", lw=1.0, ls="--")
    ax.grid(True, ls=":", lw=0.6, color="0.85")
    ax.set_xlim(float(time_ps.min()), float(time_ps.max()))
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Normalized switching coordinate", fontsize=12)
    ax.set_title("BaTiO3 switching trajectory", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 0.60))

    out_path = output_dir / "batio3_switching_trace.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Annotated BaTiO3 hysteresis trajectory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for snapshot outputs. Defaults to a sibling 'snapshot_report' folder next to the trajectory.",
    )
    parser.add_argument(
        "--rotation",
        default="90x,0y,0z",
        help="ASE rotation string used for the snapshot panel.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve() if args.input else resolve_default_input()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_path.parent / "snapshot_report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, time_ps, efield, pol, temp = load_trajectory(input_path)
    mapping = build_ti_o_mapping(frames[0])
    ti_offset_z, ti_offset_mag = compute_ti_offcentering(frames, mapping, axis=2)
    selected = choose_representative_frames(time_ps, efield, pol)
    rows = build_snapshot_rows(selected, time_ps, efield, pol, temp, ti_offset_z, ti_offset_mag)

    write_snapshot_structures(frames, rows, output_dir)
    csv_path = save_snapshot_manifest(rows, output_dir)
    panel_path = make_snapshot_panel(frames, rows, output_dir, rotation=args.rotation)
    trace_path = make_switching_trace(time_ps, efield, pol, ti_offset_z, rows, output_dir)

    print(f"Input trajectory: {input_path}")
    print(f"Saved snapshot manifest: {csv_path}")
    print(f"Saved snapshot panel: {panel_path}")
    print(f"Saved switching trace: {trace_path}")


if __name__ == "__main__":
    main()
