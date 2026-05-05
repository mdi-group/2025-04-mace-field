#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ht_dielectric_from_xyz.py  (streaming/append version)
----------------------------------------------------
High-throughput dielectric constants from MACE + MACEField.

This version writes each processed structure to the OUTPUT file immediately
(append mode), so partial results are preserved if the run stops early.

Given an input XYZ containing many structures, this script:
  - Uses a MACEField model to obtain clamped-ion susceptibility χ ("polarizability") and BECs.
  - Obtains the Γ-point Hessian from either:
      * a plain MACE model,
      * a MACE-MP-0 foundation model (optionally with D3 dispersion), or
      * the same MACEField model (if your calculator supports Hessian at E=0).
  - Supports DIFFERENT HEADS for field and Hessian models.
  - Computes ε∞, Δε_ion, and ε(0) assuming the model's "polarizability" is χ (dimensionless).
  - Immediately appends results to OUTPUT.xyz in atoms.info as flattened 9-vectors:
      MACE_eps_inf, MACE_eps_ion, MACE_eps_static
    Plus summary scalars:
      MACE_trace_inf, MACE_trace_ion, MACE_trace_static
  - Optionally also stores principal values (eigenvalues).
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from ase.io import iread, write
from ase import Atoms
from mace.calculators import MACECalculator, mace_mp
from mace.tools import torch_tools

# ---- Physical constants (SI) ----
EPS0 = 8.8541878128e-12     # F/m
E_CHG = 1.602176634e-19     # C
EV    = 1.602176634e-19     # J
ANG   = 1e-10               # m

# Conversions
EV_PER_A2_to_J_PER_M2 = EV / (ANG**2)  # 1 eV/Å^2  -> J/m^2


def reshape_alpha(alpha_raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(alpha_raw, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return arr.reshape(-1)[-9:].reshape(3, 3)


def reshape_becs(becs_raw: np.ndarray, natoms: int) -> np.ndarray:
    arr = np.asarray(becs_raw, dtype=float)
    if arr.shape == (natoms, 3, 3):
        return arr
    if arr.shape == (natoms, 9):
        return arr.reshape(natoms, 3, 3)
    if arr.size == natoms * 9:
        return arr.reshape(natoms, 3, 3)
    raise ValueError(f"Unexpected BECs shape {arr.shape}, expected (N,3,3) or (N,9)")


def assemble_Z(becs_e: np.ndarray) -> np.ndarray:
    """Arrange Z* into (3N, 3): rows are (x,y,z) per atom; columns are field α."""
    N = becs_e.shape[0]
    Z = np.zeros((3 * N, 3), dtype=float)
    for k in range(N):
        Z[3 * k + 0, :] = becs_e[k][:, 0]
        Z[3 * k + 1, :] = becs_e[k][:, 1]
        Z[3 * k + 2, :] = becs_e[k][:, 2]
    return Z


def symmetric_pseudoinverse(C_SI: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    C = 0.5 * (C_SI + C_SI.T)
    w, V = np.linalg.eigh(C)
    wmax = np.max(np.abs(w)) if w.size else 0.0
    eps = tol * (wmax if wmax > 0 else 1.0)
    w_inv = np.where(w > eps, 1.0 / w, 0.0)
    C_pinv = (V * w_inv) @ V.T
    return 0.5 * (C_pinv + C_pinv.T)


def dielectric_from(alpha_chi: np.ndarray,
                    becs_e: np.ndarray,
                    hessian_eV_per_A2: np.ndarray,
                    volume_A3: float):
    """Return (eps_inf, eps_ion, eps_static) with α interpreted as χ (dimensionless)."""
    # ε∞ = I + χ
    eps_inf = np.eye(3) + alpha_chi

    # Z*: e -> C; C: eV/Å² -> J/m²; Ω: Å³ -> m³
    Z_C = assemble_Z(becs_e * E_CHG)                       # C
    C_SI = hessian_eV_per_A2 * EV_PER_A2_to_J_PER_M2       # J/m²
    Omega = volume_A3 * (ANG ** 3)                         # m³

    # Ionic susceptibility χ_ion
    C_pinv = symmetric_pseudoinverse(C_SI, tol=1e-10)
    chi_ion = (Z_C.T @ C_pinv @ Z_C) / (EPS0 * Omega)

    # Symmetrize
    eps_inf = 0.5 * (eps_inf + eps_inf.T)
    eps_ion = 0.5 * (chi_ion + chi_ion.T)
    eps_static = 0.5 * (eps_inf + eps_ion + (eps_inf + eps_ion).T)
    return eps_inf, eps_ion, eps_static


def coerce_hessian_to_cartesian_matrix(H_raw: np.ndarray, natoms: int) -> np.ndarray:
    """
    Accept several layouts and return a (3N, 3N) Hessian.
    Supported:
      - (3N, 3N)
      - (N, N, 3, 3)  -> block to (3N, 3N)
      - (3N, N, 3)    -> flatten last two to (3N, 3N)
    """
    H = np.asarray(H_raw)
    N = natoms
    d = 3 * N
    if H.shape == (d, d):
        return H
    if H.ndim == 4 and H.shape == (N, N, 3, 3):
        return np.transpose(H, (0, 2, 1, 3)).reshape(d, d)
    if H.ndim == 3 and H.shape == (d, N, 3):
        return H.reshape(d, N * 3)
    raise ValueError(f"Unsupported Hessian shape {H.shape}; expected (3N,3N), (N,N,3,3) or (3N,N,3).")


def build_calculators(field_model_path: str,
                      hessian_model_path: str | None,
                      field_head: str,
                      hessian_head: str,
                      device: str,
                      dtype: str,
                      electric_field: list[float],
                      use_field_for_hessian: bool,
                      use_mace_mp_d3: bool,
                      mace_mp_model: str):
    """
    Build the calculators once for throughput.
    - field_calc: MACEField with chosen electric field and head (for χ and BECs)
    - hess_calc:  one of
         * MACE-MP-0 foundation model (optionally with D3),
         * a plain MACE model (.pt),
         * or MACEField@E=0 (if use_field_for_hessian=True).
    """
    import torch

    # Field-aware calculator (MACEField)
    field_calc = MACECalculator(
        model_paths=field_model_path,
        model_type="MACEField",
        device=device,
        default_dtype=dtype,
        head=field_head,
        electric_field=torch.tensor(electric_field, dtype=torch.get_default_dtype()),
    )

    # Hessian calculator selection
    if use_mace_mp_d3:
        if use_field_for_hessian:
            raise SystemExit("Cannot use both --use-field-for-hessian and --use-mace-mp-d3; choose one.")
        # MACE-MP-0 foundation model with D3 dispersion
        hess_calc = MACECalculator(
            model_paths=hessian_model_path,          # e.g. "small", "medium", "medium-mpa-0", "large"
            dispersion=True,              # include D3 dispersion correction
            default_dtype=dtype,
            device=device,
            head=hessian_head,
        )
    elif use_field_for_hessian:
        # Use field model for Hessian but force E=0 in the call site; allow a different head
        hess_calc = MACECalculator(
            model_paths=field_model_path,
            model_type="MACEField",
            device=device,
            default_dtype=dtype,
            head=hessian_head,
            electric_field=torch.zeros(3, dtype=torch.get_default_dtype()),
        )
    else:
        if hessian_model_path is None:
            raise SystemExit(
                "Provide --hessian-model, or pass --use-field-for-hessian, or use --use-mace-mp-d3."
            )
        # Plain MACE model from a .pt file
        hess_calc = MACECalculator(
            model_paths=hessian_model_path,
            model_type="MACE",
            device=device,
            default_dtype=dtype,
            head=hessian_head,
        )

    return field_calc, hess_calc


def get_hessian_direct(hess_calc, atoms: Atoms) -> np.ndarray:
    """
    Get Γ-point Hessian as (3N,3N).

    Prefer the official ASE-style API: calc.get_hessian(atoms=atoms),
    which works for MACECalculator and mace_mp (including dispersion=True).
    Fall back to the internal model call if needed.
    """
    # Preferred path: ASE calculator API
    if hasattr(hess_calc, "get_hessian"):
        H_raw = hess_calc.get_hessian(atoms=atoms)
        return coerce_hessian_to_cartesian_matrix(H_raw, natoms=len(atoms))

    # Fallback for bare MACECalculator (should rarely be needed now)
    if isinstance(hess_calc, MACECalculator):
        batch = hess_calc._atoms_to_batch(atoms)
        batch = hess_calc._clone_batch(batch)
        H_list = []
        for model in hess_calc.models:
            out = model(
                batch.to_dict(),
                compute_force=True,
                compute_stress=False,
                compute_hessian=True,
                training=hess_calc.use_compile,
            )
            H = out.get("hessian", None)
            if H is None:
                raise RuntimeError("Model did not return a Hessian (out['hessian'] is None).")
            H_list.append(H.detach().cpu().numpy())
        H_avg = np.mean(np.stack(H_list, axis=0), axis=0) if len(H_list) > 1 else H_list[0]
        return coerce_hessian_to_cartesian_matrix(H_avg, natoms=len(atoms))

    # raise RuntimeError(
    #     "Hessian calculator does not support get_hessian and is not a MACECalculator."
    # )


def compute_dielectrics_for_atoms(atoms: Atoms,
                                  field_calc,
                                  hess_calc,
                                  write_principal: bool = False):
    """
    Evaluate χ, BECs, Hessian, then compute ε tensors.
    Returns (eps_inf, eps_ion, eps_static) and also mutates atoms.info in-place.
    """
    # Attach field calc and trigger properties
    atoms.calc = field_calc
    _ = atoms.get_potential_energy()

    # Fetch & reshape
    alpha = reshape_alpha(field_calc.results["polarizability"])  # χ (dimensionless)
    becs = reshape_becs(field_calc.results["becs"], len(atoms))  # e

    # Hessian (Γ) from chosen Hessian calculator
    atoms.calc = hess_calc
    H = get_hessian_direct(hess_calc, atoms)                     # eV/Å²

    # Dielectric tensors
    eps_inf, eps_ion, eps_static = dielectric_from(alpha, becs, H, atoms.get_volume())

    # Store flattened tensors and summaries
    atoms.info["MACE_eps_inf"] = eps_inf.reshape(9)
    atoms.info["MACE_eps_ion"] = eps_ion.reshape(9)
    atoms.info["MACE_eps_static"] = eps_static.reshape(9)
    atoms.info["MACE_trace_inf"] = float(np.trace(eps_inf))
    atoms.info["MACE_trace_ion"] = float(np.trace(eps_ion))
    atoms.info["MACE_trace_static"] = float(np.trace(eps_static))

    if write_principal:
        w_inf, v_inf = np.linalg.eigh(0.5 * (eps_inf + eps_inf.T))
        w_ion, v_ion = np.linalg.eigh(0.5 * (eps_ion + eps_ion.T))
        w_sta, v_sta = np.linalg.eigh(0.5 * (eps_static + eps_static.T))
        atoms.info["MACE_principal_inf"] = w_inf
        atoms.info["MACE_principal_ion"] = w_ion
        atoms.info["MACE_principal_static"] = w_sta
    return eps_inf, eps_ion, eps_static


def set_field_info(atoms, key, vec):
    atoms.info[key] = np.asarray([float(vec[0]), float(vec[1]), float(vec[2])])


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "High-throughput dielectric tensors from MACE + MACEField "
            "for structures in an XYZ (streaming/append)."
        )
    )
    p.add_argument("input", help="Input XYZ with many structures")
    p.add_argument("output", help="Output XYZ (appended to as we go)")
    p.add_argument("--field-model", required=True,
                   help="Path to MACEField model (.pt)")
    p.add_argument("--field-head", default="Default",
                   help="Head for field model (χ and BECs), e.g. 'pt_head'")
    p.add_argument("--hessian-model", required=False, default=None,
                   help="Path to plain MACE model (.pt) for Hessian "
                        "(ignored if --use-mace-mp-d3 is set)")
    p.add_argument("--hessian-head", default="Default",
                   help="Head for Hessian model (Γ-point Hessian)")
    p.add_argument("--use-field-for-hessian", action="store_true",
                   help="Use the field model to compute Hessian (assumes E=0 path is valid)")
    p.add_argument("--use-mace-mp-d3", action="store_true",
                   help="Use MACE-MP-0 foundation model with D3 dispersion for the Hessian "
                        "(via mace_mp(..., dispersion=True))")
    p.add_argument("--mace-mp-model", default="medium",
                   help="MACE-MP-0 model string for Hessian if --use-mace-mp-d3 "
                        "(e.g. 'small', 'medium', 'medium-mpa-0', 'large')")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device")
    p.add_argument("--electric-field", nargs=3, type=float,
                   default=[0.0, 0.0, 0.0], metavar=("Ex", "Ey", "Ez"),
                   help="Macroscopic electric field used for the field-model property call (V/Å)")
    p.add_argument("--electric-field-key", default="REF_electric_field",
                   help="atoms.info key to store the macroscopic field (overwritten per frame)")
    p.add_argument("--principal", action="store_true",
                   help="Also store principal values (eigenvalues) of tensors")
    p.add_argument("--dtype", help="set default dtype",
                   type=str, choices=["float32", "float64"], default="float32")
    p.add_argument("--start", type=int, default=0,
                   help="Skip the first N frames from INPUT (useful when resuming)")
    p.add_argument("--append-output", action="store_true",
                   help="Append to OUTPUT if it exists; otherwise overwrite")
    p.add_argument("--max", type=int, default=None,
                   help="Limit number of frames processed (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.dtype)

    # Prepare output file: truncate unless --append-output is specified
    if os.path.exists(args.output) and not args.append_output:
        os.remove(args.output)

    # Build calculators once
    field_calc, hess_calc = build_calculators(
        field_model_path=args.field_model,
        hessian_model_path=args.hessian_model,
        field_head=args.field_head,
        hessian_head=args.hessian_head,
        device=args.device,
        dtype=args.dtype,
        electric_field=args.electric_field,
        use_field_for_hessian=args.use_field_for_hessian,
        use_mace_mp_d3=args.use_mace_mp_d3,
        mace_mp_model=args.mace_mp_model,
    )

    processed = 0
    failures = 0
    total_in = 0

    # Stream input frames with iread to avoid loading everything
    for idx, atoms in enumerate(iread(args.input, index=":")):
        if idx < args.start:
            continue
        if args.max is not None and processed >= args.max:
            break

        total_in += 1
        try:
            # overwrite per-frame field in atoms.info
            set_field_info(atoms, args.electric_field_key, args.electric_field)

            compute_dielectrics_for_atoms(
                atoms, field_calc, hess_calc, write_principal=args.principal
            )
            # Write this frame immediately (append)
            write(args.output, atoms, append=True)
            processed += 1
        except Exception as e:
            # Mark the error and still write the frame for traceability
            atoms.info["MACE_error"] = f"{type(e).__name__}: {e}"
            write(args.output, atoms, append=True)
            failures += 1

    print(
        f"Done. Input frames visited: {total_in}, Written: {processed + failures}, "
        f"Success: {processed}, Failures: {failures}."
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
