#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread, read, write
from ase.optimize import BFGS
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator

HERE = Path(__file__).resolve().parent
MACE_ROOT = Path.home() / "repositories" / "mace" / "mace-field"
if str(MACE_ROOT) not in sys.path:
    sys.path.insert(0, str(MACE_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mace.calculators import MACECalculator  # noqa: E402
from mace.tools import torch_tools  # noqa: E402

from ionic_dielectric_from_hessians import (  # noqa: E402
    coerce_hessian_to_cartesian_matrix,
    compare_ionic_dielectric_methods,
    ionic_dielectric_from,
    reshape_becs,
)

IONIC_DENSITY_BINS = 140
IONIC_SCATTER_THRESHOLD = 15000
IONIC_SCATTER_MAX_POINTS = 20000


DEFAULT_MODEL = HERE / "MACEField-omat-dielectric.model"
DEFAULT_HEAD = "pt_head"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute local-model Hessians, filter them, and derive ionic dielectric predictions."
    )
    parser.add_argument("input", help="Input XYZ (multi-frame)")
    parser.add_argument("output_xyz", help="Output XYZ with geometries + metadata")
    parser.add_argument("output_h5", help="Output HDF5 with Hessians")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--head", default=DEFAULT_HEAD)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA device ids to expose, e.g. '0' or '1,2'.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--maxsteps", type=int, default=300)
    parser.add_argument("--disable-cueq", action="store_true")
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--filtered-xyz", type=Path, default=None)
    parser.add_argument("--filtered-h5", type=Path, default=None)
    parser.add_argument("--eigval-tol", type=float, default=1e-5)
    parser.add_argument("--asr-tol", type=float, default=1e-4)
    parser.add_argument("--bec-asr-tol", type=float, default=0.05)
    parser.add_argument("--ionic-output-xyz", type=Path, default=None)
    parser.add_argument("--filtered-ionic-output-xyz", type=Path, default=None)
    parser.add_argument("--principal", action="store_true")
    parser.add_argument("--debug-mass-weighted", action="store_true")
    parser.add_argument("--debug-n", type=int, default=5)
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


def build_calculator(args):
    torch_tools.set_default_dtype(args.dtype)
    return MACECalculator(
        model_paths=str(args.model),
        model_type="MACEField",
        default_dtype=args.dtype,
        device=args.device,
        head=args.head,
        enable_cueq=(args.device == "cuda" and not args.disable_cueq),
    )


def maybe_relax_atoms(atoms, calc, head: str, fmax_tol: float, maxsteps: int):
    atoms = atoms.copy()
    atoms.info["head"] = head
    atoms.calc = calc
    forces = atoms.get_forces()
    f_before = float(np.abs(forces).max())
    if f_before <= fmax_tol:
        return atoms, False, f_before, f_before
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=fmax_tol, steps=maxsteps)
    forces_after = atoms.get_forces()
    f_after = float(np.abs(forces_after).max())
    return atoms, True, f_before, f_after


def compute_hessians(args):
    calc = build_calculator(args)
    if os.path.exists(args.output_xyz) and not args.append:
        os.remove(args.output_xyz)
    h5_mode = "a" if args.append else "w"
    with h5py.File(args.output_h5, h5_mode) as h5f:
        n_processed = 0
        n_fail = 0
        n_total = 0
        for idx, atoms in enumerate(iread(args.input, index=":")):
            if args.max is not None and n_processed >= args.max:
                break
            n_total += 1
            try:
                if args.relax:
                    atoms, did_relax, f_before, f_after = maybe_relax_atoms(
                        atoms,
                        calc,
                        args.head,
                        fmax_tol=args.fmax,
                        maxsteps=args.maxsteps,
                    )
                else:
                    atoms = atoms.copy()
                    atoms.info["head"] = args.head
                    atoms.calc = calc
                    forces = atoms.get_forces()
                    f_before = float(np.abs(forces).max())
                    f_after = f_before
                    did_relax = False
                force_converged = bool(np.isfinite(f_after) and f_after <= args.fmax)

                atoms.info["MACE_relaxed"] = bool(did_relax)
                atoms.info["MACE_fmax_before"] = float(f_before)
                atoms.info["MACE_fmax_after"] = float(f_after)
                atoms.info["MACE_force_converged"] = force_converged
                atoms.info["head"] = args.head

                _ = atoms.get_potential_energy()
                H_raw = calc.get_hessian(atoms=atoms)
                H = coerce_hessian_to_cartesian_matrix(H_raw, natoms=len(atoms))
                H = 0.5 * (H + H.T)

                grp = h5f.create_group(f"frame_{idx:06d}")
                grp.create_dataset("hessian_eV_per_A2", data=H, compression="gzip")
                grp.attrs["natoms"] = len(atoms)
                grp.attrs["formula"] = atoms.get_chemical_formula()
                grp.attrs["fmax_before"] = f_before
                grp.attrs["fmax_after"] = f_after
                grp.attrs["relaxed"] = bool(did_relax)
                grp.attrs["force_converged"] = force_converged

                write(args.output_xyz, atoms, append=True)
                n_processed += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                atoms = atoms.copy()
                atoms.info["MACE_Hessian_error"] = f"{type(exc).__name__}: {exc}"
                write(args.output_xyz, atoms, append=True)
                n_fail += 1
        print(
            f"Done. Frames visited: {n_total}, success: {n_processed}, failures: {n_fail}."
        )


def load_hessians_from_h5(h5_filename: str):
    hess_list = []
    meta = {}
    frame_keys = []
    with h5py.File(h5_filename, "r") as f:
        frame_keys = sorted(k for k in f.keys() if k.startswith("frame_"))
        for idx, key in enumerate(frame_keys):
            grp = f[key]
            H = grp["hessian_eV_per_A2"][...]
            hess_list.append(H)
            meta[idx] = {
                "natoms": int(grp.attrs["natoms"]),
                "formula": grp.attrs.get("formula", ""),
                "fmax_before": float(grp.attrs.get("fmax_before", np.nan)),
                "fmax_after": float(grp.attrs.get("fmax_after", np.nan)),
                "relaxed": bool(grp.attrs.get("relaxed", False)),
                "force_converged": bool(grp.attrs.get("force_converged", False)),
            }
    return hess_list, meta, frame_keys


def min_eigenvalue(H: np.ndarray) -> float:
    Hs = 0.5 * (H + H.T)
    return float(np.min(np.linalg.eigvalsh(Hs)))


def asr_violation(H: np.ndarray) -> float:
    row_sums = np.sum(H, axis=1)
    return float(np.max(np.abs(row_sums)))


def bec_asr_violation_local(becs: np.ndarray) -> float:
    arr = np.asarray(becs, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 9:
        arr = arr.reshape(-1, 3, 3)
    elif arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected REF_becs shape {arr.shape}")
    return float(np.max(np.abs(np.sum(arr, axis=0))))


def filter_dataset(input_xyz, input_h5, output_xyz, output_h5, eigval_tol, asr_tol, bec_asr_tol, fmax_tol):
    atoms_list = read(str(input_xyz), index=":")
    hessians, meta, frame_keys = load_hessians_from_h5(str(input_h5))
    if len(atoms_list) != len(hessians):
        raise RuntimeError("XYZ and HDF5 frame counts do not match.")

    if os.path.exists(output_xyz):
        os.remove(output_xyz)
    if os.path.exists(output_h5):
        os.remove(output_h5)

    rows = []
    with h5py.File(str(input_h5), "r") as fin, h5py.File(str(output_h5), "w") as fout:
        new_idx = 0
        for i, atoms in enumerate(atoms_list):
            H = np.asarray(hessians[i])
            lam_min = min_eigenvalue(H)
            h_asr = asr_violation(H)
            b_asr = bec_asr_violation_local(atoms.arrays["REF_becs"]) if "REF_becs" in atoms.arrays else np.nan
            f_before = float(meta[i]["fmax_before"])
            f_after = float(meta[i]["fmax_after"])
            relaxed = bool(meta[i]["relaxed"])
            force_converged = bool(np.isfinite(f_after) and f_after <= fmax_tol)
            keep = force_converged and lam_min >= -eigval_tol and h_asr <= asr_tol and b_asr <= bec_asr_tol
            rows.append(
                {
                    "frame_index": i,
                    "formula": atoms.get_chemical_formula(),
                    "relaxed": relaxed,
                    "force_converged": force_converged,
                    "fmax_before": f_before,
                    "fmax_after": f_after,
                    "min_eig": lam_min,
                    "hessian_asr": h_asr,
                    "bec_asr": b_asr,
                    "keep": keep,
                }
            )
            if keep:
                write(str(output_xyz), atoms, append=True)
                src_name = frame_keys[i]
                dst_name = f"frame_{new_idx:06d}"
                fin.copy(src_name, fout, name=dst_name)
                fout[dst_name].attrs["orig_frame_index"] = int(i)
                new_idx += 1
    return pd_from_rows(rows)


def pd_from_rows(rows):
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows


def write_frame_table(table, path: Path):
    if hasattr(table, "to_csv"):
        table.to_csv(path, index=False)
    else:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else [])
            writer.writeheader()
            writer.writerows(table)


def count_kept_frames(table) -> int:
    if hasattr(table, "columns"):
        if "keep" not in table.columns:
            return 0
        return int(np.count_nonzero(table["keep"].to_numpy(dtype=bool)))
    return int(sum(bool(row.get("keep", False)) for row in table))


def make_hessian_plots(h5_path: Path, plots_dir: Path, prefix: str):
    hessians, meta, _ = load_hessians_from_h5(str(h5_path))
    if not hessians:
        print(f"No Hessian frames found in {h5_path}; skipping {prefix} Hessian plots.")
        return False

    min_eigs = np.array([min_eigenvalue(H) for H in hessians], dtype=float)
    asr_errs = np.array([asr_violation(H) for H in hessians], dtype=float)
    f_before = np.array([meta[i]["fmax_before"] for i in meta], dtype=float)
    f_after = np.array([meta[i]["fmax_after"] for i in meta], dtype=float)

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.hist(min_eigs, bins=100, alpha=0.8)
    ax.set_xlabel(r"min eigenvalue of Hessian (eV/$\AA^2$)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix} Hessian minimum-eigenvalue distribution")
    ax.axvline(0.0, linestyle="--", color="0.3")
    fig.savefig(plots_dir / f"{prefix}_min_eigs.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    counts, _, _ = ax.hist(asr_errs[np.isfinite(asr_errs)], bins=100, alpha=0.8)
    if np.any(counts > 0):
        ax.set_yscale("log")
    ax.set_xlabel(r"max |row-sum(H)| (eV/$\AA^2$)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix} Hessian ASR violation")
    fig.savefig(plots_dir / f"{prefix}_hessian_asr.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    valid_force = np.isfinite(f_before) & np.isfinite(f_after)
    if np.any(valid_force):
        ax.scatter(f_before[valid_force], f_after[valid_force], s=10, alpha=0.5)
        lo = min(np.nanmin(f_before[valid_force]), np.nanmin(f_after[valid_force]))
        hi = max(np.nanmax(f_before[valid_force]), np.nanmax(f_after[valid_force]))
        ax.plot([lo, hi], [lo, hi], "--", color="0.3")
    else:
        ax.text(0.5, 0.5, "No finite force data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("fmax before (eV/Å)")
    ax.set_ylabel("fmax after (eV/Å)")
    ax.set_title(f"{prefix} relaxation summary")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig(plots_dir / f"{prefix}_fmax_scatter.png", dpi=250)
    plt.close(fig)
    return True


def compute_ionic_from_files(
    input_xyz: Path,
    input_h5: Path,
    output_xyz: Path,
    principal: bool,
    debug_mass_weighted: bool,
    debug_n: int,
):
    atoms_list = read(str(input_xyz), index=":")
    if len(atoms_list) == 0:
        print(f"No frames found in {input_xyz}; skipping ionic dielectric calculation.")
        return 0
    hessians, _, _ = load_hessians_from_h5(str(input_h5))
    if output_xyz.exists():
        output_xyz.unlink()

    debug_done = 0
    out_atoms = []
    for idx, atoms in enumerate(atoms_list):
        H = np.asarray(hessians[idx], dtype=float)
        becs = reshape_becs(np.asarray(atoms.arrays["REF_becs"], dtype=float), len(atoms))
        volume = float(abs(atoms.get_volume()))
        eps_ion = ionic_dielectric_from(becs, H, volume)

        out = atoms.copy()
        out.info["eps_ion"] = eps_ion.reshape(9)
        out.info["trace_eps_ion"] = float(np.trace(eps_ion))
        out.info["eps_static_ion"] = out.info["eps_ion"]
        if principal:
            w, _ = np.linalg.eigh(0.5 * (eps_ion + eps_ion.T))
            out.info["principal_eps_ion"] = w

        if debug_mass_weighted and debug_done < debug_n:
            masses = np.asarray(atoms.get_masses(), dtype=float)
            eps_direct, eps_mass, abs_norm, rel_norm = compare_ionic_dielectric_methods(
                becs,
                H,
                volume,
                masses,
            )
            out.info["eps_ion_massweighted"] = eps_mass.reshape(9)
            out.info["eps_ion_rel_diff"] = float(rel_norm)
            out.info["eps_ion_abs_diff"] = float(abs_norm)
            debug_done += 1
        out_atoms.append(out)
    write(str(output_xyz), out_atoms)
    return len(out_atoms)


def _load_ref_pred_from_xyz(xyz_path: Path, ref_key: str, pred_key: str):
    atoms_list = read(str(xyz_path), index=":")
    ref_list = []
    pred_list = []
    for at in atoms_list:
        if ref_key not in at.info or pred_key not in at.info:
            continue
        ref_list.append(np.asarray(at.info[ref_key], dtype=float).ravel())
        pred_list.append(np.asarray(at.info[pred_key], dtype=float).ravel())
    if not ref_list:
        empty = np.empty((0, 3, 3), dtype=float)
        return empty, empty
    REF = np.vstack(ref_list).reshape(-1, 3, 3)
    PRED = np.vstack(pred_list).reshape(-1, 3, 3)
    return REF, PRED


def plot_ionic_dielectric_parity_2panel(xyz_path: Path, plots_dir: Path, prefix: str, ref_key="REF_epsilon_ionic", pred_key="eps_ion"):
    REF, PRED = _load_ref_pred_from_xyz(xyz_path, ref_key, pred_key)
    if REF.size == 0 or PRED.size == 0:
        print(f"No ionic dielectric pairs found in {xyz_path}; skipping {prefix} parity plot.")
        return False
    diag_ref = np.concatenate([REF[:, i, i] for i in range(3)])
    diag_pred = np.concatenate([PRED[:, i, i] for i in range(3)])
    off_pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    off_ref = np.concatenate([REF[:, i, j] for i, j in off_pairs])
    off_pred = np.concatenate([PRED[:, i, j] for i, j in off_pairs])

    def metrics(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0 or y.size == 0:
            return x, y, {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "slope": np.nan, "intercept": np.nan}
        resid = y - x
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae = float(np.mean(np.abs(resid)))
        if x.size >= 2 and float(np.ptp(x)) > 0.0:
            m, c = np.polyfit(x, y, 1)
            yhat = m * x + c
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            m, c, r2 = np.nan, np.nan, np.nan
        return x, y, {"rmse": rmse, "mae": mae, "r2": r2, "slope": float(m), "intercept": float(c)}

    fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.2), constrained_layout=True)
    for ax, raw_x, raw_y, title in (
        (axs[0], diag_ref, diag_pred, "Diagonals"),
        (axs[1], off_ref, off_pred, "Off-diagonals"),
    ):
        x, y, stat = metrics(raw_x, raw_y)
        if x.size > 0 and y.size > 0:
            vals = np.r_[x, y]
            lo, hi = np.percentile(vals, [1, 99])
            if title == "Diagonals":
                lim = (0.0, max(hi, 0.0) * 1.05 if hi > 0 else 1.0)
            else:
                lim_abs = max(abs(lo), abs(hi))
                lim = (-1.05 * lim_abs, 1.05 * lim_abs if lim_abs > 0 else 1.0)
            plot_mask = (x >= lim[0]) & (x <= lim[1]) & (y >= lim[0]) & (y <= lim[1])
            plot_x = x[plot_mask]
            plot_y = y[plot_mask]
            if plot_x.size <= IONIC_SCATTER_THRESHOLD:
                if plot_x.size > IONIC_SCATTER_MAX_POINTS:
                    sample_idx = np.linspace(0, plot_x.size - 1, IONIC_SCATTER_MAX_POINTS, dtype=int)
                    plot_x = plot_x[sample_idx]
                    plot_y = plot_y[sample_idx]
                ax.scatter(plot_x, plot_y, s=8, alpha=0.28, c="tab:blue", edgecolors="none")
                density_artist = None
            else:
                _, _, _, density_artist = ax.hist2d(
                    plot_x,
                    plot_y,
                    bins=IONIC_DENSITY_BINS,
                    range=[lim, lim],
                    norm=LogNorm(),
                    cmin=1,
                    cmap="viridis",
                )
            ax.plot(lim, lim, "--", color="0.3", lw=1.2)
            if np.isfinite(stat["slope"]):
                xx = np.linspace(*lim, 200)
                ax.plot(xx, stat["slope"] * xx + stat["intercept"], color="tab:orange", lw=1.4)
            ax.text(
                0.02,
                0.98,
                f"R²={stat['r2']:.3f}\nRMSE={stat['rmse']:.4g}\nMAE={stat['mae']:.4g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
            )
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            if density_artist is not None:
                cb = fig.colorbar(density_artist, ax=ax, pad=0.02)
                cb.set_label("log density")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("Reference ionic dielectric")
        ax.set_ylabel("Predicted ionic dielectric")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.suptitle(f"{prefix} ionic dielectric parity")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{prefix}_ionic_dielectric_parity.png", dpi=450)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    apply_visible_gpus(args.gpus)
    if not args.relax and (args.ionic_output_xyz is not None or args.filtered_ionic_output_xyz is not None):
        print(
            "Warning: ionic dielectric tensors are being computed from unrelaxed structures. "
            "This often leaves soft residual modes and can seriously degrade ionic-dielectric parity."
        )
    compute_hessians(args)

    plots_dir = args.plots_dir or (Path(args.output_xyz).resolve().parent / "plots")
    make_hessian_plots(Path(args.output_h5), plots_dir, "unfiltered")

    filter_table = None
    filtered_kept = 0
    if args.filtered_xyz is not None and args.filtered_h5 is not None:
        filter_table = filter_dataset(
            Path(args.output_xyz),
            Path(args.output_h5),
            args.filtered_xyz,
            args.filtered_h5,
            args.eigval_tol,
            args.asr_tol,
            args.bec_asr_tol,
            args.fmax,
        )
        write_frame_table(filter_table, plots_dir / "filter_summary.csv")
        filtered_kept = count_kept_frames(filter_table)
        print(f"Filtered Hessian frames kept: {filtered_kept} / {len(filter_table)}")
        if filtered_kept > 0:
            make_hessian_plots(args.filtered_h5, plots_dir, "filtered")
        else:
            print("Filtered Hessian dataset is empty; skipping filtered Hessian plots and ionic dielectric analysis.")

    if args.ionic_output_xyz is not None:
        compute_ionic_from_files(
            Path(args.output_xyz),
            Path(args.output_h5),
            args.ionic_output_xyz,
            principal=args.principal,
            debug_mass_weighted=args.debug_mass_weighted,
            debug_n=args.debug_n,
        )
        plot_ionic_dielectric_parity_2panel(args.ionic_output_xyz, plots_dir, "unfiltered")

    if (
        filtered_kept > 0
        and args.filtered_ionic_output_xyz is not None
        and args.filtered_xyz is not None
        and args.filtered_h5 is not None
    ):
        compute_ionic_from_files(
            args.filtered_xyz,
            args.filtered_h5,
            args.filtered_ionic_output_xyz,
            principal=args.principal,
            debug_mass_weighted=args.debug_mass_weighted,
            debug_n=args.debug_n,
        )
        plot_ionic_dielectric_parity_2panel(args.filtered_ionic_output_xyz, plots_dir, "filtered")

    print(f"Hessian workflow complete. Outputs: {args.output_xyz}, {args.output_h5}")


if __name__ == "__main__":
    main()
