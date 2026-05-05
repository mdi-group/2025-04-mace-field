#!/usr/bin/env python3
"""Create representative alpha-quartz MD snapshots and thermo diagnostics.

This script selects representative frames from an annotated SiO2 MD trajectory,
writes the structures to disk, and produces a snapshot montage plus a thermal
diagnostic figure suitable for Supplementary Information.

Example
-------
python make_spectroscopy_snapshots.py \
  --input ../LAMMPs/MD/runs/SiO2-.../SiO2-mp-7000/production.annotated.extxyz
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


def nearest_image_vector(cell, origin, target):
    frac = np.linalg.solve(cell.T, (target - origin).T).T
    frac -= np.round(frac)
    return frac @ cell


def resolve_default_input() -> Path:
    runs_root = Path(__file__).resolve().parents[1] / "LAMMPs" / "MD" / "runs"
    candidates = sorted(
        runs_root.glob("SiO2-*/SiO2-mp-7000/production.annotated.extxyz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not find any SiO2 production.annotated.extxyz trajectories.")
    return candidates[0]


def load_trajectory(path: Path):
    frames = read(str(path), ":")
    if not frames:
        raise ValueError(f"No frames found in {path}")
    time_ps = np.asarray([float(at.info.get("time_ps", i)) for i, at in enumerate(frames)], dtype=float)
    temp = np.asarray([float(at.info.get("temperature_K", np.nan)) for at in frames], dtype=float)
    total_energy = np.asarray([float(at.info.get("lammps_total_energy", np.nan)) for at in frames], dtype=float)
    pressure = np.asarray([float(at.info.get("lammps_pressure_bar", np.nan)) for at in frames], dtype=float)
    pol_norm = np.asarray(
        [np.linalg.norm(np.asarray(at.info["MACE_polarization"], dtype=float)) for at in frames],
        dtype=float,
    )
    return frames, time_ps, temp, total_energy, pressure, pol_norm


def choose_representative_frames(time_ps, temp):
    target_times_ps = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]
    selected = []
    used = set()
    for target in target_times_ps:
        idx = int(np.argmin(np.abs(np.asarray(time_ps, dtype=float) - target)))
        if idx in used:
            continue
        used.add(idx)
        selected.append(idx)
    selected.sort()
    return selected


def build_rows(indices, time_ps, temp, total_energy, pressure, pol_norm):
    rows = []
    for order, idx in enumerate(indices, start=1):
        label = f"frame_{order}"
        rows.append(
            {
                "order": order,
                "label": label,
                "frame_index": int(idx),
                "time_ps": float(time_ps[idx]),
                "temperature_K": float(temp[idx]),
                "total_energy_eV": float(total_energy[idx]),
                "pressure_bar": float(pressure[idx]),
                "polarization_norm_e_A2": float(pol_norm[idx]),
            }
        )
    return rows


def parse_rotation(rotation: str):
    angles = {"x": 0.0, "y": 0.0, "z": 0.0}
    for chunk in rotation.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        axis = chunk[-1].lower()
        value = float(chunk[:-1])
        if axis not in angles:
            raise ValueError(f"Unsupported rotation chunk {chunk!r}")
        angles[axis] = value
    return angles["x"], angles["y"], angles["z"]


def rotated_copy(at, rotation: str):
    rx, ry, rz = parse_rotation(rotation)
    rotated = at.copy()
    rotated.rotate(rx, "x", center=(0.0, 0.0, 0.0), rotate_cell=True)
    rotated.rotate(ry, "y", center=(0.0, 0.0, 0.0), rotate_cell=True)
    rotated.rotate(rz, "z", center=(0.0, 0.0, 0.0), rotate_cell=True)
    return rotated


def cell_edges_2d(cell):
    a, b, c = np.asarray(cell, dtype=float)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            a,
            b,
            c,
            a + b,
            a + c,
            b + c,
            a + b + c,
        ]
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return corners[:, :2], edges


def build_bond_segments(at, cutoff=2.05):
    symbols = np.asarray(at.symbols)
    si_indices = np.where(symbols == "Si")[0]
    o_indices = np.where(symbols == "O")[0]
    positions = at.get_positions()
    cell = np.asarray(at.cell)
    segments = []
    for si_idx in si_indices:
        for o_idx in o_indices:
            rel = nearest_image_vector(cell, positions[si_idx], positions[o_idx])
            dist = np.linalg.norm(rel)
            if dist <= cutoff:
                start = positions[si_idx]
                end = positions[si_idx] + rel
                segments.append((start[:2], end[:2]))
    return segments


def plot_2d_snapshot(ax, at, rotation: str):
    rotated = rotated_copy(at, rotation)
    coords = rotated.get_positions()[:, :2]
    symbols = np.asarray(at.symbols)
    colors = {"Si": "#d8a031", "O": "#d63b2f"}
    sizes = {"Si": 90, "O": 58}
    corners_2d, edges = cell_edges_2d(rotated.cell)
    for i, j in edges:
        ax.plot(
            [corners_2d[i, 0], corners_2d[j, 0]],
            [corners_2d[i, 1], corners_2d[j, 1]],
            color="0.55",
            lw=0.8,
            zorder=0,
        )
    for start, end in build_bond_segments(rotated):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="0.40",
            lw=1.0,
            alpha=0.9,
            zorder=1,
        )
    for symbol in ("O", "Si"):
        mask = symbols == symbol
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=sizes[symbol],
            c=colors[symbol],
            edgecolors="black",
            linewidths=0.35,
            zorder=3 if symbol == "Si" else 2,
        )
    all_xy = np.vstack([coords, corners_2d])
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_snapshot_structures(frames, rows, output_dir: Path):
    snapshot_dir = output_dir / "structures"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for old_file in snapshot_dir.glob("*.xyz"):
        old_file.unlink()
    for row in rows:
        at = frames[row["frame_index"]].copy()
        filename = snapshot_dir / f"{row['order']:02d}_{row['label']}.xyz"
        write(filename, at)
        row["structure_path"] = str(filename)


def save_snapshot_manifest(rows, output_dir: Path):
    csv_path = output_dir / "sio2_snapshot_summary.csv"
    fieldnames = [
        "order",
        "label",
        "frame_index",
        "time_ps",
        "temperature_K",
        "total_energy_eV",
        "pressure_bar",
        "polarization_norm_e_A2",
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        at = frames[row["frame_index"]]
        plot_2d_snapshot(ax, at, rotation=rotation)
        ax.set_title(
            (
                f"t = {row['time_ps']:.1f} ps, "
                f"T = {row['temperature_K']:.1f} K\n"
                f"P = {row['pressure_bar']:.0f} bar"
            ),
            fontsize=10,
        )
    for ax in axes[len(rows) :]:
        ax.set_axis_off()
    fig.tight_layout()
    out_path = output_dir / "sio2_representative_snapshots.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_thermo_trace(time_ps, temp, total_energy, rows, output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)

    axes[0].plot(time_ps, temp, color="tab:red", lw=1.8)
    for row in rows:
        axes[0].scatter(row["time_ps"], row["temperature_K"], color="tab:red", s=28, zorder=5)
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(r"$\alpha$-SiO$_2$ trajectory diagnostics")
    axes[0].grid(True, ls=":", lw=0.6, color="0.85")

    axes[1].plot(time_ps, total_energy, color="tab:blue", lw=1.8)
    for row in rows:
        axes[1].scatter(row["time_ps"], row["total_energy_eV"], color="tab:blue", s=28, zorder=5)
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Total energy (eV)")
    axes[1].grid(True, ls=":", lw=0.6, color="0.85")

    fig.tight_layout()
    out_path = output_dir / "sio2_thermo_trace.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Annotated SiO2 production trajectory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to a sibling 'snapshot_report' folder next to the trajectory.",
    )
    parser.add_argument(
        "--rotation",
        default="75x,18y,12z",
        help="Rotation string for the 2D projected snapshot panel, e.g. '75x,18y,12z'.",
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

    frames, time_ps, temp, total_energy, pressure, pol_norm = load_trajectory(input_path)
    indices = choose_representative_frames(time_ps, temp)
    rows = build_rows(indices, time_ps, temp, total_energy, pressure, pol_norm)

    write_snapshot_structures(frames, rows, output_dir)
    csv_path = save_snapshot_manifest(rows, output_dir)
    panel_path = make_snapshot_panel(frames, rows, output_dir, rotation=args.rotation)
    trace_path = make_thermo_trace(time_ps, temp, total_energy, rows, output_dir)

    print(f"Input trajectory: {input_path}")
    print(f"Saved snapshot manifest: {csv_path}")
    print(f"Saved snapshot panel: {panel_path}")
    print(f"Saved thermo trace: {trace_path}")


if __name__ == "__main__":
    main()
