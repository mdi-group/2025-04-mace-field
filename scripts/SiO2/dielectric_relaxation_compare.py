#!/usr/bin/env python3
"""Compare SiO2 finite-field relaxation dielectric runs.

This is a small companion to the Allegro-pol quartz dielectric-relaxation
workflow. It reads one or more run directories containing:

- ``relaxed_zero.annotated.extxyz``
- ``relaxed_field_x.annotated.extxyz`` / ``relaxed_field_y.annotated.extxyz`` /
  ``relaxed_field_z.annotated.extxyz`` (preferred)
- or the older single-direction ``relaxed_field.annotated.extxyz`` fallback

and reports dielectric constants derived from the relaxed structures.

For each run it computes:
- electronic dielectric tensor from the zero-field polarizability
- static dielectric response along any applied field directions present
- parallel/perpendicular summaries with respect to a chosen polar axis

The static response uses a sign-corrected polarization change because the
current quartz relaxation workflow follows the opposite polarization sign
convention to the one used in the dielectric-constant formula.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

POL_UC_CM2 = 1602.176634
EPS0_CONST = 5.526349406e-3  # e * V^{-1} * A^{-1}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "a": 0, "b": 1, "c": 2}


@dataclass
class RunResult:
    label: str
    run_dir: Path
    model_name: str
    head: str | None
    eps_inf_diag: np.ndarray
    eps_0_diag: np.ndarray
    field_vectors: dict[str, np.ndarray]
    delta_polarization: dict[str, np.ndarray]
    polarization_zero: np.ndarray
    polarization_field: dict[str, np.ndarray]
    static_note: str


def axis_settings(ax):
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.0)
    ax.tick_params(which="major", width=2.2, length=9, direction="in")
    ax.tick_params(which="minor", width=1.8, length=5, direction="in")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")


def infer_label(run_dir: Path) -> str:
    files = sorted(run_dir.glob("*.model"))
    if not files:
        return run_dir.parent.name
    name = files[0].name.lower()
    if "omat" in name:
        return "omat"
    if "finetuned" in name:
        return "finetuned"
    if "sio2" in name:
        return "direct"
    return files[0].stem


def load_atoms(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return read(path)


def find_summary_json(run_dir: Path) -> Path | None:
    path = run_dir / "dielectric_relax_summary.json"
    return path if path.exists() else None


def extract_eps_inf_diag(zero_atoms) -> np.ndarray:
    alpha = np.asarray(zero_atoms.info["MACE_polarizability"], dtype=float).reshape(3, 3)
    return 1.0 + np.diag(alpha)


def find_field_files(run_dir: Path) -> dict[str, Path]:
    files = {}
    for axis in ("x", "y", "z"):
        path = run_dir / f"relaxed_field_{axis}.annotated.extxyz"
        if path.exists():
            files[axis] = path
    if files:
        return files

    fallback = run_dir / "relaxed_field.annotated.extxyz"
    if fallback.exists():
        atoms = load_atoms(fallback)
        field = np.asarray(atoms.info["MACE_electric_field"], dtype=float).reshape(3)
        axis = "xyz"[int(np.argmax(np.abs(field)))]
        return {axis: fallback}
    raise FileNotFoundError(
        f"No field-relaxed annotated extxyz files found under {run_dir}"
    )


def extract_static_eps(
    zero_atoms,
    field_atoms_by_axis: dict[str, object],
    response_sign: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    p0 = np.asarray(zero_atoms.info["MACE_polarization"], dtype=float).reshape(3)
    alpha0 = np.asarray(zero_atoms.info["MACE_polarizability"], dtype=float).reshape(3, 3)
    field_vectors: dict[str, np.ndarray] = {}
    field_polarizations: dict[str, np.ndarray] = {}
    delta_by_axis: dict[str, np.ndarray] = {}
    eps = np.full(3, np.nan, dtype=float)

    for axis, atoms in field_atoms_by_axis.items():
        pf = np.asarray(atoms.info["MACE_polarization"], dtype=float).reshape(3)
        field = np.asarray(atoms.info["MACE_electric_field"], dtype=float).reshape(3)
        idx = AXIS_INDEX[axis]
        delta_p = response_sign * (pf - p0)
        field_vectors[axis] = field
        field_polarizations[axis] = pf
        delta_by_axis[axis] = delta_p
        component = field[idx]
        if abs(component) > 1.0e-12:
            eps[idx] = 1.0 + alpha0[idx, idx] + delta_p[idx] / (EPS0_CONST * component)

    return field_vectors, p0, field_polarizations, delta_by_axis, eps


def load_run_result(label: str, run_dir: Path, response_sign: float) -> RunResult:
    zero_atoms = load_atoms(run_dir / "relaxed_zero.annotated.extxyz")
    field_files = find_field_files(run_dir)
    field_atoms_by_axis = {axis: load_atoms(path) for axis, path in field_files.items()}
    summary_path = find_summary_json(run_dir)
    summary = json.loads(summary_path.read_text()) if summary_path else {}

    field_vectors, p0, pf_by_axis, delta_by_axis, eps_0 = extract_static_eps(
        zero_atoms, field_atoms_by_axis, response_sign
    )
    eps_inf = extract_eps_inf_diag(zero_atoms)
    model_name = Path(summary.get("model_path", "")).name or infer_label(run_dir)
    head = summary.get("head")

    missing_axes = [axis for axis in ("x", "y", "z") if axis not in field_files]
    static_note = ""
    if missing_axes:
        static_note = (
            "Missing field relaxations for "
            + ", ".join(missing_axes)
            + "; perpendicular/parallel static values may be incomplete."
        )

    return RunResult(
        label=label,
        run_dir=run_dir,
        model_name=model_name,
        head=head,
        eps_inf_diag=eps_inf,
        eps_0_diag=eps_0,
        field_vectors=field_vectors,
        delta_polarization=delta_by_axis,
        polarization_zero=p0,
        polarization_field=pf_by_axis,
        static_note=static_note,
    )


def build_calc(model_path: str, head: str | None, device: str = "cuda", dtype: str = "float32"):
    from mace.calculators import MACECalculator

    kwargs = {
        "model_paths": model_path,
        "device": device,
        "default_dtype": dtype,
        "model_type": "MACEField",
        "enable_cueq": False,
        "enable_oeq": False,
    }
    if head:
        kwargs["head"] = head
    return MACECalculator(**kwargs)


def parse_lammpstrj(path: Path, symbols_by_type: dict[int, str]) -> tuple[list[np.ndarray], np.ndarray]:
    from ase import Atoms

    frames = []
    cell = None
    with path.open("r", encoding="utf-8") as handle:
        lines = iter(handle)
        for line in lines:
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            next(lines)  # timestep value
            next(lines)  # ITEM: NUMBER OF ATOMS
            natoms = int(next(lines).strip())
            next(lines)  # ITEM: BOX BOUNDS ...
            bounds = [next(lines).split() for _ in range(3)]
            lo = np.array([float(b[0]) for b in bounds], dtype=float)
            hi = np.array([float(b[1]) for b in bounds], dtype=float)
            cell = hi - lo
            header = next(lines).strip().split()[2:]
            idx_id = header.index("id")
            idx_type = header.index("type")
            idx_x = header.index("xu")
            idx_y = header.index("yu")
            idx_z = header.index("zu")

            ids = np.empty(natoms, dtype=int)
            types = np.empty(natoms, dtype=int)
            pos = np.empty((natoms, 3), dtype=float)
            for i in range(natoms):
                parts = next(lines).split()
                ids[i] = int(parts[idx_id])
                types[i] = int(parts[idx_type])
                pos[i, 0] = float(parts[idx_x])
                pos[i, 1] = float(parts[idx_y])
                pos[i, 2] = float(parts[idx_z])
            order = np.argsort(ids)
            symbols = [symbols_by_type[int(t)] for t in types[order]]
            atoms = Atoms(symbols=symbols, positions=pos[order], cell=cell, pbc=True)
            frames.append(atoms)
    return frames, np.asarray(cell, dtype=float)


def compute_relaxation_curve(
    run_dir: Path,
    axis: str,
    model_name: str,
    head: str | None,
    response_sign: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    zero_traj = run_dir / "relax_zero.lammpstrj"
    field_traj = run_dir / f"relaxed_field_{axis}.lammpstrj"
    if not zero_traj.exists() or not field_traj.exists():
        return None

    model_path = run_dir / model_name
    if not model_path.exists():
        return None

    zero_frames, _ = parse_lammpstrj(zero_traj, {1: "O", 2: "Si"})
    field_frames, _ = parse_lammpstrj(field_traj, {1: "O", 2: "Si"})
    if not zero_frames or not field_frames:
        return None

    calc = build_calc(str(model_path), head=head)
    calc.electric_field = np.zeros(3)
    zero_relaxed = zero_frames[-1]
    p0 = np.asarray(calc.get_property("polarization", zero_relaxed), dtype=float).reshape(3)
    alpha0 = np.asarray(calc.get_property("polarizability", zero_relaxed), dtype=float).reshape(3, 3)

    axis_index = AXIS_INDEX[axis]
    steps = np.arange(len(field_frames) + 1, dtype=int)
    eps_values = np.empty(len(field_frames) + 1, dtype=float)
    field_component = float(
        read(run_dir / f"relaxed_field_{axis}.annotated.extxyz").info["MACE_electric_field"][axis_index]
    )
    eps_values[0] = 1.0 + alpha0[axis_index, axis_index]
    for i, atoms in enumerate(field_frames):
        field_vec = np.zeros(3, dtype=float)
        field_vec[axis_index] = field_component
        calc.electric_field = field_vec
        pf = np.asarray(calc.get_property("polarization", atoms), dtype=float).reshape(3)
        alpha = np.asarray(calc.get_property("polarizability", atoms), dtype=float).reshape(3, 3)
        delta = response_sign * (pf[axis_index] - p0[axis_index])
        eps_values[i + 1] = 1.0 + alpha[axis_index, axis_index] + delta / (EPS0_CONST * field_component)
    return steps, eps_values


def plot_relaxation(
    results: Iterable[RunResult],
    output_prefix: Path,
    polar_axis: str,
    response_sign: float,
) -> Path | None:
    curves = []
    for result in results:
        parallel_curve = compute_relaxation_curve(
            result.run_dir,
            axis=polar_axis,
            model_name=result.model_name,
            head=result.head,
            response_sign=response_sign,
        )
        perp_curves = []
        for axis in ("x", "y", "z"):
            if axis == polar_axis:
                continue
            curve = compute_relaxation_curve(
                result.run_dir,
                axis=axis,
                model_name=result.model_name,
                head=result.head,
                response_sign=response_sign,
            )
            if curve is not None:
                perp_curves.append(curve)
        if parallel_curve is not None or perp_curves:
            curves.append((result, parallel_curve, perp_curves))

    if not curves:
        return None

    fig, axes = plt.subplots(
        1, len(curves), figsize=(5.6 * len(curves), 4.8), dpi=180, squeeze=False
    )
    axes = axes[0]
    color_cycle = ["#0055d4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for ax, (result, parallel_curve, perp_curves), color in zip(axes, curves, color_cycle):
        axis_settings(ax)
        ymin = np.inf
        ymax = -np.inf

        if parallel_curve is not None:
            steps_para, eps_para = parallel_curve
            ax.plot(
                steps_para,
                eps_para,
                marker="o",
                ms=4.8,
                lw=1.8,
                color=color,
                label="parallel",
            )
            ax.scatter([steps_para[0], steps_para[-1]], [eps_para[0], eps_para[-1]], c=color, s=38, zorder=5)
            ymin = min(ymin, float(np.min(eps_para)))
            ymax = max(ymax, float(np.max(eps_para)))

        if perp_curves:
            min_len = min(len(steps) for steps, _ in perp_curves)
            steps_perp = perp_curves[0][0][:min_len]
            eps_perp = np.mean([eps[:min_len] for _, eps in perp_curves], axis=0)
            ax.plot(
                steps_perp,
                eps_perp,
                marker="s",
                ms=4.2,
                lw=1.6,
                color="0.25",
                label="perpendicular",
            )
            ax.scatter([steps_perp[0], steps_perp[-1]], [eps_perp[0], eps_perp[-1]], c="0.25", s=34, zorder=5)
            ymin = min(ymin, float(np.min(eps_perp)))
            ymax = max(ymax, float(np.max(eps_perp)))

        x_text = 0.04 * max(
            [parallel_curve[0][-1] if parallel_curve is not None else 0]
            + [curve[0][-1] for curve in perp_curves]
            + [1]
        )
        if parallel_curve is not None:
            eps_inf_para, eps_0_para = parallel_perpendicular(
                np.array([np.nan, np.nan, result.eps_inf_diag[AXIS_INDEX[polar_axis]]]), 2
            )
            eps0_para_text = result.eps_0_diag[AXIS_INDEX[polar_axis]]
            ax.text(
                x_text,
                parallel_curve[1][0],
                f"ε∞,|| = {parallel_curve[1][0]:.2f}",
                color=color,
                va="center",
            )
            ax.text(
                parallel_curve[0][-1] * 0.68 if parallel_curve[0][-1] > 0 else 0.2,
                parallel_curve[1][-1] * 0.995,
                f"ε₀,|| = {eps0_para_text:.2f}",
                color=color,
                va="top",
            )
        if perp_curves:
            ax.text(
                x_text,
                eps_perp[0],
                f"ε∞,⊥ = {eps_perp[0]:.2f}",
                color="0.25",
                va="center",
            )
            eps0_perp = parallel_perpendicular(result.eps_0_diag, AXIS_INDEX[polar_axis])[1]
            if eps0_perp is not None:
                ax.text(
                    steps_perp[-1] * 0.68 if steps_perp[-1] > 0 else 0.2,
                    eps_perp[-1] * 1.005,
                    f"ε₀,⊥ = {eps0_perp:.2f}",
                    color="0.25",
                    va="bottom",
                )

        if np.isfinite(ymin) and np.isfinite(ymax):
            pad = max(0.15, 0.08 * (ymax - ymin))
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(result.label)
        ax.set_xlabel("relax steps")
        ax.set_ylabel("ε")
        ax.legend(frameon=False, loc="best")

    fig.suptitle(f"SiO2 dielectric relaxation ({polar_axis}-axis reference)", y=1.02)
    fig.tight_layout()
    path = output_prefix.with_name(output_prefix.name + "_relaxation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parallel_perpendicular(diag: np.ndarray, axis_index: int) -> tuple[float | None, float | None]:
    parallel = None if np.isnan(diag[axis_index]) else float(diag[axis_index])
    perp_values = [diag[i] for i in range(3) if i != axis_index and not np.isnan(diag[i])]
    perpendicular = float(np.mean(perp_values)) if perp_values else None
    return parallel, perpendicular


def write_summary(
    results: Iterable[RunResult],
    output_prefix: Path,
    polar_axis: str,
    response_sign: float,
) -> tuple[Path, Path]:
    axis_index = AXIS_INDEX[polar_axis]
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    rows = []
    for result in results:
        eps_inf_para, eps_inf_perp = parallel_perpendicular(result.eps_inf_diag, axis_index)
        eps_0_para, eps_0_perp = parallel_perpendicular(result.eps_0_diag, axis_index)

        row = {
            "label": result.label,
            "run_dir": str(result.run_dir),
            "model_name": result.model_name,
            "head": result.head,
            "available_field_axes": " ".join(sorted(result.field_vectors)),
            "eps_inf_x": float(result.eps_inf_diag[0]),
            "eps_inf_y": float(result.eps_inf_diag[1]),
            "eps_inf_z": float(result.eps_inf_diag[2]),
            "eps_inf_parallel": eps_inf_para,
            "eps_inf_perpendicular": eps_inf_perp,
            "eps_0_x": None if np.isnan(result.eps_0_diag[0]) else float(result.eps_0_diag[0]),
            "eps_0_y": None if np.isnan(result.eps_0_diag[1]) else float(result.eps_0_diag[1]),
            "eps_0_z": None if np.isnan(result.eps_0_diag[2]) else float(result.eps_0_diag[2]),
            "eps_0_parallel": eps_0_para,
            "eps_0_perpendicular": eps_0_perp,
            "P0_x_uC_cm2": float(result.polarization_zero[0] * POL_UC_CM2),
            "P0_y_uC_cm2": float(result.polarization_zero[1] * POL_UC_CM2),
            "P0_z_uC_cm2": float(result.polarization_zero[2] * POL_UC_CM2),
            "deltaP_x_uC_cm2": float(result.delta_polarization.get("x", np.zeros(3))[0] * POL_UC_CM2),
            "deltaP_y_uC_cm2": float(result.delta_polarization.get("y", np.zeros(3))[1] * POL_UC_CM2),
            "deltaP_z_uC_cm2": float(result.delta_polarization.get("z", np.zeros(3))[2] * POL_UC_CM2),
            "static_note": result.static_note,
        }
        rows.append(row)

    payload = {
        "polar_axis": polar_axis,
        "eps0_const_e_per_VA": EPS0_CONST,
        "polarization_response_sign": response_sign,
        "note": (
            "eps_inf is derived from I + MACE_polarizability at zero field. "
            "eps_0 is derived from sign-corrected delta polarization using the available directional relaxations."
        ),
        "runs": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, json_path


def print_console_summary(results: Iterable[RunResult], polar_axis: str) -> None:
    axis_index = AXIS_INDEX[polar_axis]
    print(f"Polar axis: {polar_axis}")
    print("")
    for result in results:
        eps_inf_para, eps_inf_perp = parallel_perpendicular(result.eps_inf_diag, axis_index)
        eps_0_para, eps_0_perp = parallel_perpendicular(result.eps_0_diag, axis_index)
        print(f"{result.label} ({result.model_name})")
        print(f"  run: {result.run_dir}")
        print(f"  available field axes: {' '.join(sorted(result.field_vectors))}")
        print(
            "  eps_inf: "
            f"parallel={eps_inf_para:.4f} "
            f"perpendicular={eps_inf_perp:.4f}"
        )
        eps0_perp_text = "unavailable" if eps_0_perp is None else f"{eps_0_perp:.4f}"
        eps0_para_text = "unavailable" if eps_0_para is None else f"{eps_0_para:.4f}"
        print(
            "  eps_0:   "
            f"parallel={eps0_para_text} "
            f"perpendicular={eps0_perp_text}"
        )
        if result.static_note:
            print(f"  note: {result.static_note}")
        print("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize SiO2 finite-field relaxation dielectric constants from MACEField runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        help="Run label and directory containing relaxed_zero plus directional relaxed_field*.annotated.extxyz files. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--polar-axis",
        default="z",
        choices=sorted(AXIS_INDEX),
        help="Polar axis for parallel/perpendicular summaries. Default: z.",
    )
    parser.add_argument(
        "--response-sign",
        type=float,
        default=-1.0,
        help="Multiplier applied to delta polarization before converting to static epsilon. Default: -1.",
    )
    parser.add_argument(
        "--output-prefix",
        default="/home/brad/repositories/2025-04-mace-field/scripts/SiO2/plots/sio2_dielectric_relax_compare",
        help="Prefix for output CSV/JSON summary files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_entries = args.run or [
        [
            "omat",
            "/home/brad/repositories/2025-04-mace-field/scripts/LAMMPs/MD/runs/SiO2-mp-7000-sc1x1x1-dielectric-relax-2026-05-04_150526/SiO2-mp-7000",
        ],
        [
            "direct",
            "/home/brad/repositories/2025-04-mace-field/scripts/LAMMPs/MD/runs/SiO2-mp-7000-sc1x1x1-dielectric-relax-2026-05-04_150533/SiO2-mp-7000",
        ],
    ]
    results = [
        load_run_result(label, Path(path).expanduser().resolve(), args.response_sign)
        for label, path in run_entries
    ]

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = write_summary(results, output_prefix, args.polar_axis, args.response_sign)
    plot_path = plot_relaxation(results, output_prefix, args.polar_axis, args.response_sign)
    print_console_summary(results, args.polar_axis)
    print(f"Wrote CSV summary: {csv_path}")
    print(f"Wrote JSON summary: {json_path}")
    if plot_path is not None:
        print(f"Wrote relaxation plot: {plot_path}")
    else:
        print("Relaxation plot not written: no stepwise relaxation trajectories were found.")


if __name__ == "__main__":
    main()
