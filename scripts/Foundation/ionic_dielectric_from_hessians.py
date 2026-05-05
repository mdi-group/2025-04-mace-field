#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ionic_dielectric_from_hessians.py
---------------------------------
Compute ionic dielectric tensors Δε_ion from:

  - Hessians stored in an HDF5 file (per frame), and
  - Born effective charges (BECs) stored in an XYZ file (per frame).

Now also has an optional debug path that computes Δε_ion via
a mass-weighted dynamical-matrix route and compares the two.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import h5py
from ase.io import iread, write
from ase import Atoms

# ---- Physical constants (SI) ----
EPS0 = 8.8541878128e-12     # F/m
E_CHG = 1.602176634e-19     # C
EV    = 1.602176634e-19     # J
ANG   = 1e-10               # m

# Conversions
EV_PER_A2_to_J_PER_M2 = EV / (ANG ** 2)  # 1 eV/Å^2  -> J/m^2


def reshape_becs(becs_raw: np.ndarray, natoms: int) -> np.ndarray:
    """Coerce BECs to shape (N, 3, 3)."""
    arr = np.asarray(becs_raw, dtype=float)
    if arr.shape == (natoms, 3, 3):
        return arr
    if arr.shape == (natoms, 9):
        return arr.reshape(natoms, 3, 3)
    if arr.size == natoms * 9:
        return arr.reshape(natoms, 3, 3)
    raise ValueError(f"Unexpected BECs shape {arr.shape}, expected (N,3,3) or (N,9)")


def assemble_Z(becs_e: np.ndarray) -> np.ndarray:
    """
    Arrange Z* into (3N, 3): rows are (x,y,z) per atom; columns are field α.

    becs_e: shape (N, 3, 3), in units of e.
      - index 0: atom index κ
      - index 1: polarisation component i
      - index 2: field component α
    """
    N = becs_e.shape[0]
    Z = np.zeros((3 * N, 3), dtype=float)
    for k in range(N):
        Z[3 * k + 0, :] = becs_e[k][:, 0]
        Z[3 * k + 1, :] = becs_e[k][:, 1]
        Z[3 * k + 2, :] = becs_e[k][:, 2]
    return Z


def enforce_bec_charge_neutrality(becs_e: np.ndarray) -> np.ndarray:
    """
    Enforce the BEC acoustic sum rule by removing the per-component mean charge
    tensor so translational zero modes do not couple to a spurious net charge.
    """
    becs = np.asarray(becs_e, dtype=float)
    return becs - np.mean(becs, axis=0, keepdims=True)


def translational_projector(natoms: int) -> np.ndarray:
    """
    Projector that removes the three rigid-translation directions from a 3N
    Cartesian displacement space.
    """
    d = 3 * natoms
    basis = []
    for cart in range(3):
        vec = np.zeros(d, dtype=float)
        vec[cart::3] = 1.0
        vec /= np.linalg.norm(vec)
        basis.append(vec)
    projector = np.eye(d, dtype=float)
    for vec in basis:
        projector -= np.outer(vec, vec)
    return projector


def symmetric_pseudoinverse(C_SI: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Symmetric Moore–Penrose pseudoinverse for a (possibly singular) symmetric matrix.
    """
    C = 0.5 * (C_SI + C_SI.T)
    w, V = np.linalg.eigh(C)
    wmax = np.max(np.abs(w)) if w.size else 0.0
    eps = tol * (wmax if wmax > 0 else 1.0)
    w_inv = np.where(w > eps, 1.0 / w, 0.0)
    C_pinv = (V * w_inv) @ V.T
    return 0.5 * (C_pinv + C_pinv.T)


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


# ---------- direct method: Z^T C^+ Z ----------

def ionic_dielectric_from(
    becs_e: np.ndarray,
    hessian_eV_per_A2: np.ndarray,
    volume_A3: float,
) -> np.ndarray:
    """
    Compute ionic contribution to the dielectric tensor Δε_ion (3x3)
    via the direct force-constant inverse: χ_ion = Z^T C^+ Z / (ε0 Ω).
    """
    becs_e = enforce_bec_charge_neutrality(becs_e)
    N = becs_e.shape[0]
    # Z*: e -> C
    Z_C = assemble_Z(becs_e * E_CHG)                   # (3N,3) in C
    # C: eV/Å² -> J/m²
    C_SI = hessian_eV_per_A2 * EV_PER_A2_to_J_PER_M2   # (3N,3N) in J/m²
    # Ω: Å³ -> m³
    Omega = volume_A3 * (ANG ** 3)                     # m³

    # Pseudoinverse of the dynamical matrix
    C_pinv = symmetric_pseudoinverse(C_SI, tol=1e-10)
    C_pinv = translational_projector(N) @ C_pinv @ translational_projector(N)

    # Ionic susceptibility χ_ion
    chi_ion = (Z_C.T @ C_pinv @ Z_C) / (EPS0 * Omega)

    # Symmetrise to enforce χ_ion = χ_ion^T
    eps_ion = 0.5 * (chi_ion + chi_ion.T)
    return eps_ion


# ---------- mass-weighted route: phonon dynamical matrix ----------

def ionic_dielectric_from_massweighted(
    becs_e: np.ndarray,
    hessian_eV_per_A2: np.ndarray,
    volume_A3: float,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """
    Same ionic dielectric, but via the mass-weighted dynamical matrix:

        D = M^{-1/2} C M^{-1/2}
        Z_tilde = M^{-1/2} Z

        χ_ion = Z_tilde^T D^+ Z_tilde / (ε0 Ω)

    and Δε_ion = χ_ion.
    """
    becs_e = enforce_bec_charge_neutrality(becs_e)
    natoms = becs_e.shape[0]
    assert masses_amu.shape == (natoms,)

    d = 3 * natoms
    # mass vector repeated for x,y,z of each atom
    M_vec = np.repeat(masses_amu, 3)            # (3N,)
    Minv_sqrt = 1.0 / np.sqrt(M_vec)           # (3N,)

    # force constants C in SI
    C_SI = hessian_eV_per_A2 * EV_PER_A2_to_J_PER_M2   # (3N,3N)
    # dynamical matrix
    D = (Minv_sqrt[:, None] * C_SI * Minv_sqrt[None, :])

    D_pinv = symmetric_pseudoinverse(D, tol=1e-10)
    D_pinv = translational_projector(natoms) @ D_pinv @ translational_projector(natoms)

    # mass-weighted effective charges
    Z_C = assemble_Z(becs_e * E_CHG)           # (3N,3)
    Z_tilde = Minv_sqrt[:, None] * Z_C         # (3N,3)

    Omega = volume_A3 * (ANG ** 3)

    chi_ion = (Z_tilde.T @ D_pinv @ Z_tilde) / (EPS0 * Omega)
    eps_ion = 0.5 * (chi_ion + chi_ion.T)
    return eps_ion


def compare_ionic_dielectric_methods(
    becs_e: np.ndarray,
    hessian_eV_per_A2: np.ndarray,
    volume_A3: float,
    masses_amu: np.ndarray,
):
    """
    Compute Δε_ion via both methods and return:
        eps_direct, eps_mass, abs_diff_norm, rel_diff_norm
    where norms are Frobenius norms.
    """
    eps_direct = ionic_dielectric_from(becs_e, hessian_eV_per_A2, volume_A3)
    eps_mass   = ionic_dielectric_from_massweighted(becs_e, hessian_eV_per_A2,
                                                    volume_A3, masses_amu)

    diff = eps_mass - eps_direct
    abs_norm = float(np.linalg.norm(diff))
    base_norm = float(np.linalg.norm(eps_direct))
    rel_norm = abs_norm / (base_norm + 1e-16)
    return eps_direct, eps_mass, abs_norm, rel_norm


# ---------- CLI plumbing ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute ionic dielectric tensors from Hessians (.h5) and BECs (.xyz)."
    )
    p.add_argument("input_xyz", help="Input XYZ with frames and BECs (e.g. REF_becs)")
    p.add_argument("input_h5", help="Input HDF5 with Hessians (e.g. dielectric_hessians.h5)")
    p.add_argument("output_xyz", help="Output XYZ with ionic dielectric tensors in atoms.info")
    p.add_argument(
        "--becs-key",
        default="REF_becs",
        help="atoms.arrays key containing BECs (default: REF_becs)",
    )
    p.add_argument(
        "--principal",
        action="store_true",
        help="Also store principal values (eigenvalues) of Δε_ion",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N frames (useful for resuming)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limit number of frames processed (debug)",
    )
    p.add_argument(
        "--append-output",
        action="store_true",
        help="Append to OUTPUT if it exists; otherwise overwrite",
    )
    p.add_argument(
        "--debug-mass-weighted",
        action="store_true",
        help="For first few frames, also compute Δε_ion via the mass-weighted "
             "dynamical-matrix route and print differences.",
    )
    p.add_argument(
        "--debug-n",
        type=int,
        default=5,
        help="Number of frames for which to run the mass-weighted comparison (default: 5).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare output XYZ
    if os.path.exists(args.output_xyz) and not args.append_output:
        os.remove(args.output_xyz)

    processed = 0
    failures = 0
    total_in = 0
    debug_done = 0

    with h5py.File(args.input_h5, "r") as h5f:
        # We assume groups are named frame_000000, frame_000001, ...
        for idx, atoms in enumerate(iread(args.input_xyz, index=":")):
            if idx < args.start:
                continue
            if args.max is not None and processed >= args.max:
                break

            total_in += 1
            frame_name = f"frame_{idx:06d}"

            try:
                if frame_name not in h5f:
                    raise KeyError(f"Group {frame_name} not found in {args.input_h5}")

                grp = h5f[frame_name]
                H_raw = grp["hessian_eV_per_A2"][...]       # eV/Å^2

                natoms = len(atoms)
                H = coerce_hessian_to_cartesian_matrix(H_raw, natoms=natoms)

                if args.becs_key not in atoms.arrays:
                    raise KeyError(
                        f"atoms.arrays['{args.becs_key}'] not found for frame {idx}"
                    )

                becs_raw = atoms.arrays[args.becs_key]
                becs = reshape_becs(becs_raw, natoms)

                # Ionic dielectric tensor (direct method)
                eps_ion = ionic_dielectric_from(
                    becs_e=becs,
                    hessian_eV_per_A2=H,
                    volume_A3=atoms.get_volume(),
                )

                # Optional debug comparison with mass-weighted route
                if args.debug_mass_weighted and debug_done < args.debug_n:
                    masses = atoms.get_masses()
                    eps_direct, eps_mass, abs_norm, rel_norm = \
                        compare_ionic_dielectric_methods(
                            becs_e=becs,
                            hessian_eV_per_A2=H,
                            volume_A3=atoms.get_volume(),
                            masses_amu=masses,
                        )
                    print(
                        f"[DEBUG frame {idx}] |Δε_ion|_F = {abs_norm:.3e}, "
                        f"relative = {rel_norm:.3e}"
                    )
                    # (Optional) store mass-weighted result & relative diff
                    atoms.info["eps_ion_massweighted"] = eps_mass.reshape(9)
                    atoms.info["eps_ion_rel_diff"] = rel_norm
                    debug_done += 1

                # Store in atoms.info
                atoms.info["eps_ion"] = eps_ion.reshape(9)
                atoms.info["trace_eps_ion"] = float(np.trace(eps_ion))
                # alias for convenience
                atoms.info["eps_static_ion"] = atoms.info["eps_ion"]

                if args.principal:
                    w, v = np.linalg.eigh(0.5 * (eps_ion + eps_ion.T))
                    atoms.info["principal_eps_ion"] = w

                write(args.output_xyz, atoms, append=True)
                processed += 1

            except Exception as e:
                atoms.info["dielectric_error"] = f"{type(e).__name__}: {e}"
                write(args.output_xyz, atoms, append=True)
                failures += 1

    print(
        f"Done. Input frames visited: {total_in}, "
        f"Written: {processed + failures}, "
        f"Success: {processed}, Failures: {failures}."
    )
    print(f"Output: {args.output_xyz}")


if __name__ == "__main__":
    main()
