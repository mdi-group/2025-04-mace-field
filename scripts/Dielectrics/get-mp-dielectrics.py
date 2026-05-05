#!/usr/bin/env python3
"""
Collect MP materials with DFPT dielectric/BEC data and write a combined
multi-frame .extxyz using the dielectric task itself for structure, forces,
stress, energy, dielectric tensors, and BECs.

This version PATCHES stress to be stored in ASE/MACE units: eV/Å^3.

Requirements:
  pip install mp-api ase pymatgen numpy tqdm

Usage:
  export MP_API_KEY="YOUR_KEY"
  python get-mp-dielectrics.py --out MP-Dielectrics.extxyz --write-filtered --dedupe-filtered --write-splits

Notes:
- We use the dielectric endpoint as a proxy for "has DFPT dielectric task" (and often BECs).
- We map materials to dielectric tasks via SummaryDoc.origins.
- We batch task fetches to <=100 (TaskRester commonly enforces small batch sizes).
- We stream-write extxyz frames to avoid holding everything in memory.
- Optional post-processing can filter, deduplicate, and split the dataset for training.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ase.io import read as ase_read
from ase.io import write as ase_write
from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ---------------------------
# Stress unit conversion
# ---------------------------
# We want REF_stress in eV/Å^3 (ASE/MACE convention).
# VASP prints stress in kB (kbar). 1 kbar = 0.1 GPa.
# 1 eV/Å^3 = 160.21766208 GPa  => 1 GPa = 1/160.21766208 eV/Å^3
GPA_PER_EV_A3 = 160.21766208
EV_A3_PER_GPA = 1.0 / GPA_PER_EV_A3
EV_A3_PER_KBAR = 0.1 * EV_A3_PER_GPA  # (kbar -> GPa -> eV/Å^3)

# Set this depending on what you believe MP TaskDoc stress contains:
# - "kbar": VASP OUTCAR convention (common)
# - "gpa" : already in GPa (less common; depends on parser/schema)
STRESS_INPUT_UNITS = "kbar"
HERE = Path(__file__).resolve().parent
LOCAL_FILTERED_DIELECTRIC_FALLBACK = (
    HERE.parent / "Foundation" / "data" / "MP-Dielectrics-and-Ferroelectrics.xyz"
)
LOCAL_RAW_DIELECTRIC_FALLBACKS = [
    Path.home() / "repositories" / "mace" / "Dielectric" / "final_data" / "mp-dielectric.extxyz",
    Path.home() / "repositories" / "mace" / "Dielectric" / "final_data" / "legacy-mp-dielectric.extxyz",
]


# ---------------------------
# Helpers
# ---------------------------

def _pick_origin_task_id(origins: Any, name: str) -> Optional[str]:
    """Return task_id string for a given PropertyOrigin name (e.g. 'dielectric', 'structure')."""
    if not origins:
        return None
    for o in origins:
        origin_name = o.get("name") if isinstance(o, dict) else getattr(o, "name", None)
        if origin_name == name:
            tid = o.get("task_id") if isinstance(o, dict) else getattr(o, "task_id", None)
            return str(tid) if tid is not None else None
    return None


def _canonical_task_id(task_id: Any) -> Optional[str]:
    """Normalize legacy MP task IDs like 'mp-1140435' to '1140435'."""
    if task_id is None:
        return None
    tid = str(task_id).strip()
    if not tid:
        return None
    if tid.startswith("mp-") and tid[3:].isdigit():
        return tid[3:]
    return tid


def _to_np(x: Any, dtype=float) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        return np.array(x, dtype=dtype)
    except Exception:
        return None


def _safe_get(obj: Any, path: Sequence[Any], default=None):
    """Duck-typed getter for nested attrs/dicts."""
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        elif isinstance(cur, (list, tuple)):
            idx = p
            if isinstance(idx, str):
                if not idx.lstrip("-").isdigit():
                    return default
                idx = int(idx)
            if not isinstance(idx, int):
                return default
            if idx >= len(cur) or idx < -len(cur):
                return default
            cur = cur[idx]
        else:
            cur = getattr(cur, str(p), None)
    return default if cur is None else cur


def _first_present(obj: Any, paths: Sequence[Sequence[Any]], default=None):
    """Return the first non-None value found across several candidate paths."""
    for path in paths:
        value = _safe_get(obj, path, default=None)
        if value is not None:
            return value
    return default


def _extract_outcar_dict(task_doc: Any) -> Optional[dict]:
    """Locate the parsed OUTCAR payload across task-doc variants."""
    outcar = _first_present(
        task_doc,
        (
            ("calcs_reversed", 0, "output", "outcar"),
            ("output", "outcar"),
            ("outcar",),
        ),
        default=None,
    )
    return outcar if isinstance(outcar, dict) else None


def _coerce_structure(structure: Any) -> Any:
    """Convert raw task-doc structure dicts back into pymatgen Structure objects."""
    if structure is None or isinstance(structure, Structure):
        return structure
    if isinstance(structure, dict):
        try:
            return Structure.from_dict(structure)
        except Exception:
            return None
    return structure


def _extract_dfpt_tensors(task_doc: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract epsilon_static, epsilon_ionic, epsilon_total, born (BECs) from a DFPT dielectric task doc.

    Common locations:
      task_doc.calcs_reversed[0].output.epsilon_static
      task_doc.calcs_reversed[0].output.epsilon_ionic
      task_doc.calcs_reversed[0].output.outcar["born"] (or a close variant)
    """
    eps_static = _to_np(
        _first_present(
            task_doc,
            (
                ("calcs_reversed", 0, "output", "epsilon_static"),
                ("output", "epsilon_static"),
                ("epsilon_static",),
            ),
            default=None,
        )
    )
    eps_ionic = _to_np(
        _first_present(
            task_doc,
            (
                ("calcs_reversed", 0, "output", "epsilon_ionic"),
                ("output", "epsilon_ionic"),
                ("epsilon_ionic",),
            ),
            default=None,
        )
    )

    outcar = _extract_outcar_dict(task_doc)
    if eps_static is None and isinstance(outcar, dict):
        eps_static = _to_np(
            outcar.get("dielectric_tensor")
            or outcar.get("epsilon_static")
            or outcar.get("epsilon_electronic")
        )
    if eps_ionic is None and isinstance(outcar, dict):
        eps_ionic = _to_np(
            outcar.get("dielectric_ionic_tensor")
            or outcar.get("epsilon_ionic")
        )

    eps_total = None
    if isinstance(outcar, dict):
        eps_total = _to_np(
            outcar.get("dielectric_total_tensor")
            or outcar.get("epsilon_total")
        )
    if (
        eps_total is None
        and eps_static is not None
        and eps_ionic is not None
        and eps_static.shape == (3, 3)
        and eps_ionic.shape == (3, 3)
    ):
        eps_total = eps_static + eps_ionic

    born = None
    if isinstance(outcar, dict):
        born = outcar.get("born") or outcar.get("born_charges") or outcar.get("born_effective_charges")
    born = _to_np(born)

    return eps_static, eps_ionic, eps_total, born


def _extract_dielectric_doc_tensors(diel_doc: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract dielectric tensors directly from the dielectric endpoint doc."""
    eps_static = _to_np(
        _first_present(
            diel_doc,
            (
                ("electronic",),
                ("epsilon_static",),
            ),
            default=None,
        )
    )
    eps_ionic = _to_np(
        _first_present(
            diel_doc,
            (
                ("ionic",),
                ("epsilon_ionic",),
            ),
            default=None,
        )
    )

    eps_total = _to_np(
        _first_present(
            diel_doc,
            (
                ("total",),
                ("epsilon_total",),
            ),
            default=None,
        )
    )
    if (
        eps_total is None
        and eps_static is not None
        and eps_ionic is not None
        and eps_static.shape == (3, 3)
        and eps_ionic.shape == (3, 3)
    ):
        eps_total = eps_static + eps_ionic

    return eps_static, eps_ionic, eps_total


def _extract_task_quantities(task_doc: Any) -> Tuple[Any, Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract structure, energy, forces, and stress from a task doc.

    Typical:
      task_doc.output.structure
      task_doc.output.energy
      task_doc.output.forces
      task_doc.output.stress
    """
    structure = _first_present(
        task_doc,
        (
            ("output", "structure"),
            ("calcs_reversed", 0, "output", "structure"),
            ("structure",),
        ),
        default=None,
    )
    energy = _first_present(
        task_doc,
        (
            ("output", "energy"),
            ("calcs_reversed", 0, "output", "energy"),
            ("energy",),
        ),
        default=None,
    )
    forces = _to_np(
        _first_present(
            task_doc,
            (
                ("output", "forces"),
                ("calcs_reversed", 0, "output", "forces"),
                ("calcs_reversed", 0, "output", "ionic_steps", -1, "forces"),
                ("forces",),
            ),
            default=None,
        )
    )
    stress = _to_np(
        _first_present(
            task_doc,
            (
                ("output", "stress"),
                ("calcs_reversed", 0, "output", "stress"),
                ("calcs_reversed", 0, "output", "ionic_steps", -1, "stress"),
                ("stress",),
            ),
            default=None,
        )
    )
    try:
        energy = float(energy) if energy is not None else None
    except Exception:
        energy = None
    return _coerce_structure(structure), energy, forces, stress


def _stress_to_ev_a3(stress: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Convert stress to eV/Å^3. Accepts 3x3 tensor or 6-vector (Voigt), returns same shape.
    Assumes STRESS_INPUT_UNITS is either "kbar" or "gpa".
    """
    if stress is None:
        return None
    s = np.array(stress, dtype=float)

    if STRESS_INPUT_UNITS.lower() == "kbar":
        factor = EV_A3_PER_KBAR
    elif STRESS_INPUT_UNITS.lower() == "gpa":
        factor = EV_A3_PER_GPA
    else:
        raise ValueError(f"Unknown STRESS_INPUT_UNITS={STRESS_INPUT_UNITS} (use 'kbar' or 'gpa')")

    return s * factor


def _stress_to_flat(stress: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Return stress as a flattened array for extxyz comment line.
    If stress is 3x3 -> flatten row-major; if Voigt 6 -> keep as-is.
    """
    if stress is None:
        return None
    s = np.array(stress, dtype=float)
    if s.shape == (3, 3):
        return s.reshape(-1)
    if s.shape == (6,):
        return s
    return s.reshape(-1)


def _write_extxyz_frame(
    fh,
    atoms,
    info: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
):
    """
    Minimal extxyz writer so we can stream frames without holding all in memory.

    Format:
      N
      key=value key="string" ...  (second line)
      Sym x y z [per-atom props...]
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    n = len(symbols)
    fh.write(f"{n}\n")

    # extxyz Lattice: 9 floats row-major
    lat = cell.array.reshape(-1)
    comment_parts = [f'Lattice="{ " ".join(f"{v:.10g}" for v in lat.tolist()) }"', 'pbc="T T T"']

    # Scalars/arrays in comment line
    for k, v in info.items():
        if v is None:
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            comment_parts.append(f"{k}={float(v):.10g}")
        else:
            arr = np.array(v).reshape(-1) if isinstance(v, (list, tuple, np.ndarray)) else None
            if arr is not None:
                comment_parts.append(f'{k}="{ " ".join(f"{x:.10g}" for x in arr.tolist()) }"')
            else:
                s = str(v).replace('"', "'")
                comment_parts.append(f'{k}="{s}"')

    # Properties declaration
    prop_fields = ["species:S:1", "pos:R:3"]
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.shape[0] != n:
            continue
        cols = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        prop_fields.append(f"{name}:R:{cols}")

    comment_parts.append(f'Properties={":".join(prop_fields)}')
    fh.write(" ".join(comment_parts) + "\n")

    # Per-atom lines
    for i in range(n):
        row = [symbols[i], f"{positions[i,0]:.10g}", f"{positions[i,1]:.10g}", f"{positions[i,2]:.10g}"]
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if arr.shape[0] != n:
                continue
            vals = arr[i].reshape(-1) if arr.ndim > 1 else np.array([arr[i]])
            row.extend(f"{float(x):.10g}" for x in vals.tolist())
        fh.write(" ".join(row) + "\n")


def _batched(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _normalize_split_ratios(ratios: Sequence[float]) -> np.ndarray:
    r = np.array(ratios, dtype=float)
    if r.shape != (3,) or np.any(r < 0):
        raise ValueError("Split ratios must be three non-negative numbers: train valid test")
    if r.sum() <= 0:
        raise ValueError("At least one split ratio must be positive")
    return r / r.sum()


def _allocate_counts(total: int, weights: Sequence[float]) -> np.ndarray:
    weights = np.array(weights, dtype=float)
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(len(weights), dtype=int)
    raw = total * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    while counts.sum() < total:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1
    return counts


def _split_counts(n_items: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    ratios = _normalize_split_ratios(ratios)
    positive = int(np.count_nonzero(ratios > 0))
    if n_items < positive:
        raise ValueError(f"Need at least {positive} items for the requested non-empty split ratios")

    raw = ratios * n_items
    counts = np.floor(raw).astype(int)
    mins = (ratios > 0).astype(int)
    counts = np.maximum(counts, mins)

    while counts.sum() > n_items:
        reducible = np.where(counts > mins)[0]
        if len(reducible) == 0:
            break
        idx = int(reducible[np.argmin(raw[reducible] - counts[reducible])])
        counts[idx] -= 1

    while counts.sum() < n_items:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def _read_atoms_list(path: Path) -> List[Any]:
    atoms = ase_read(path, index=":")
    return atoms if isinstance(atoms, list) else [atoms]


def _load_local_dielectric_fallback(prefer_filtered: bool) -> Tuple[Optional[List[Any]], Optional[str]]:
    if prefer_filtered and LOCAL_FILTERED_DIELECTRIC_FALLBACK.exists():
        atoms = _read_atoms_list(LOCAL_FILTERED_DIELECTRIC_FALLBACK)
        dielectric_atoms = [at for at in atoms if "dielectric_task_id" in at.info]
        if dielectric_atoms:
            return dielectric_atoms, str(LOCAL_FILTERED_DIELECTRIC_FALLBACK)

    for path in LOCAL_RAW_DIELECTRIC_FALLBACKS:
        if path.exists():
            atoms = _read_atoms_list(path)
            if atoms:
                return atoms, str(path)

    return None, None


def _rewrite_extxyz_with_ase(path: Path, atoms_list: Optional[List[Any]] = None) -> List[Any]:
    atoms_list = list(atoms_list) if atoms_list is not None else _read_atoms_list(path)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.stem}.",
        suffix=path.suffix or ".extxyz",
        dir=path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        ase_write(tmp_path, atoms_list, format="extxyz")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return atoms_list


def _reshape_tensor9(values: Any) -> Optional[np.ndarray]:
    arr = _to_np(values)
    if arr is None:
        return None
    flat = np.asarray(arr, dtype=float).reshape(-1)
    if flat.size != 9:
        return None
    return flat.reshape(3, 3)


def _reshape_bec_array(values: Any, natoms: int) -> Optional[np.ndarray]:
    arr = _to_np(values)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.shape == (natoms, 9):
        return arr.reshape(natoms, 3, 3)
    if arr.shape == (natoms, 3, 3):
        return arr
    return None


def _frame_eps_total(atoms: Any) -> Optional[np.ndarray]:
    eps_total = _reshape_tensor9(atoms.info.get("REF_epsilon_total"))
    if eps_total is not None:
        return eps_total
    eps_static = _reshape_tensor9(atoms.info.get("REF_epsilon_static"))
    eps_ionic = _reshape_tensor9(atoms.info.get("REF_epsilon_ionic"))
    if eps_static is not None and eps_ionic is not None:
        return eps_static + eps_ionic
    return None


def _frame_polarizability(atoms: Any) -> Optional[np.ndarray]:
    arr = _to_np(atoms.info.get("REF_polarizability"))
    if arr is None:
        return None
    flat = np.asarray(arr, dtype=float).reshape(-1)
    if flat.size != 9:
        return None
    return flat


def _frame_becs(atoms: Any) -> Optional[np.ndarray]:
    return _reshape_bec_array(atoms.arrays.get("REF_becs"), len(atoms))


def _frame_force_max(atoms: Any) -> float:
    forces = atoms.arrays.get("REF_forces")
    if forces is None:
        return float("nan")
    try:
        forces_arr = np.asarray(forces, dtype=float)
        if forces_arr.shape != (len(atoms), 3):
            return float("nan")
        return float(np.max(np.linalg.norm(forces_arr, axis=1)))
    except Exception:
        return float("nan")


def _frame_stress_tensor(atoms: Any) -> Optional[np.ndarray]:
    arr = _to_np(atoms.info.get("REF_stress"))
    if arr is None:
        return None
    flat = np.asarray(arr, dtype=float).reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3)
    if flat.size == 6:
        return flat
    return None


def _frame_stress_mag_ev_a3(atoms: Any) -> float:
    stress = _frame_stress_tensor(atoms)
    if stress is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(stress, dtype=float).reshape(-1))))


def _frame_bec_asr_norm(atoms: Any) -> float:
    bec = _frame_becs(atoms)
    if bec is None:
        return float("nan")
    return float(np.linalg.norm(np.sum(bec, axis=0)))


def _frame_eps_stats(atoms: Any) -> Tuple[Optional[np.ndarray], float, float]:
    eps_total = _frame_eps_total(atoms)
    if eps_total is None:
        return None, float("nan"), float("nan")
    try:
        eps_sym = 0.5 * (np.asarray(eps_total, dtype=float) + np.asarray(eps_total, dtype=float).T)
        eigvals = np.linalg.eigvalsh(eps_sym)
        if np.any(~np.isfinite(eigvals)):
            return eps_sym, float("nan"), float("nan")
        return eps_sym, float(np.min(eigvals)), float(np.mean(eigvals))
    except Exception:
        return eps_total, float("nan"), float("nan")


def _weight_from_info(atoms: Any, key: str, default: float = 1.0) -> float:
    try:
        value = atoms.info.get(key, default)
        return float(default if value is None else value)
    except Exception:
        return float(default)


def _filter_atoms(
    atoms_list: Sequence[Any],
    *,
    max_force_ev_a: float,
    min_eps_eig: float,
    max_eps_scalar: float,
    max_asr_norm: float,
    max_stress_ev_a3: Optional[float],
    require_bec: bool,
    require_eps_total: bool,
    require_pol: bool,
    require_stress: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    kept: List[Any] = []
    drop_reason_counts: Counter[str] = Counter()
    fail_count_hist: Counter[int] = Counter()

    for idx, atoms in enumerate(atoms_list):
        reasons: List[str] = []

        w_becs = _weight_from_info(atoms, "config_becs_weight", 1.0)
        w_stress = _weight_from_info(atoms, "config_stress_weight", 1.0)
        w_pol = _weight_from_info(atoms, "config_polarizability_weight", 1.0)

        bec = _frame_becs(atoms)
        eps_total, eps_min_eig, eps_scalar = _frame_eps_stats(atoms)
        pol = _frame_polarizability(atoms)
        max_force = _frame_force_max(atoms)
        bec_asr_norm = _frame_bec_asr_norm(atoms)
        stress_mag_ev_a3 = _frame_stress_mag_ev_a3(atoms)

        if require_bec and w_becs <= 0:
            reasons.append("disabled_bec")
        if require_stress and w_stress <= 0:
            reasons.append("disabled_stress")
        if require_pol and w_pol <= 0:
            reasons.append("disabled_polarizability")

        if require_bec and bec is None:
            reasons.append("missing_bec")
        if require_eps_total and eps_total is None:
            reasons.append("missing_eps_total")
        if require_pol and pol is None:
            reasons.append("missing_polarizability")
        if require_stress and not np.isfinite(stress_mag_ev_a3):
            reasons.append("missing_stress")

        if eps_total is not None:
            if not np.isfinite(eps_min_eig) or eps_min_eig < min_eps_eig:
                reasons.append("eps_min_eig")
            if not np.isfinite(eps_scalar) or eps_scalar <= 0 or eps_scalar > max_eps_scalar:
                reasons.append("eps_scalar")

        if np.isfinite(max_force) and max_force > max_force_ev_a:
            reasons.append("max_force")
        if np.isfinite(bec_asr_norm) and bec_asr_norm > max_asr_norm:
            reasons.append("bec_asr")
        if max_stress_ev_a3 is not None and np.isfinite(stress_mag_ev_a3) and stress_mag_ev_a3 > max_stress_ev_a3:
            reasons.append("stress")

        if reasons:
            fail_count_hist[len(set(reasons))] += 1
            drop_reason_counts.update(set(reasons))
            continue

        kept_atoms = atoms.copy()
        kept_atoms.info["filter_source_index"] = idx
        kept.append(kept_atoms)

    summary = {
        "n_input": len(atoms_list),
        "n_kept": len(kept),
        "n_dropped": len(atoms_list) - len(kept),
        "drop_reason_counts": dict(sorted(drop_reason_counts.items())),
        "fail_count_hist": dict(sorted(fail_count_hist.items())),
    }
    return kept, summary


def _dedupe_atoms(
    atoms_list: Sequence[Any],
    *,
    ltol: float,
    stol: float,
    angle_tol: float,
) -> Tuple[List[Any], Dict[str, Any]]:
    if not atoms_list:
        return [], {"n_input": 0, "n_groups": 0, "n_kept": 0}

    adaptor = AseAtomsAdaptor()
    structures = []
    structure_idx: Dict[int, int] = {}
    for idx, atoms in enumerate(atoms_list):
        structure = adaptor.get_structure(atoms)
        structures.append(structure)
        structure_idx[id(structure)] = idx

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    groups = matcher.group_structures(structures)
    keep_idx = sorted(structure_idx[id(group[0])] for group in groups if group)
    unique_atoms = [atoms_list[idx].copy() for idx in keep_idx]
    summary = {
        "n_input": len(atoms_list),
        "n_groups": len(groups),
        "n_kept": len(unique_atoms),
        "n_removed": len(atoms_list) - len(unique_atoms),
    }
    return unique_atoms, summary


def _atoms_feature_matrix(atoms_list: Sequence[Any]) -> np.ndarray:
    if not atoms_list:
        return np.zeros((0, 1), dtype=float)

    unique_z = sorted({int(z) for atoms in atoms_list for z in np.unique(atoms.numbers)})
    rows: List[np.ndarray] = []
    for atoms in atoms_list:
        numbers = np.asarray(atoms.numbers, dtype=float)
        natoms = max(len(numbers), 1)
        comp = np.array([np.count_nonzero(numbers == z) for z in unique_z], dtype=float) / natoms
        cellpar = np.asarray(atoms.cell.cellpar(), dtype=float)
        volume = float(atoms.get_volume()) if len(atoms) > 0 else 0.0
        row = np.concatenate(
            [
                comp,
                np.array(
                    [
                        np.log1p(natoms),
                        volume / natoms,
                        float(numbers.mean()),
                        float(numbers.std()),
                        cellpar[0],
                        cellpar[1],
                        cellpar[2],
                        cellpar[3] / 180.0,
                        cellpar[4] / 180.0,
                        cellpar[5] / 180.0,
                    ],
                    dtype=float,
                ),
            ]
        )
        rows.append(row)

    mat = np.vstack(rows)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    keep = std > 1e-12
    if not np.any(keep):
        return np.zeros((len(atoms_list), 1), dtype=float)
    return (mat[:, keep] - mean[keep]) / std[keep]


def _group_indices(atoms_list: Sequence[Any], group_key: str) -> Tuple[List[str], List[List[int]]]:
    groups: Dict[str, List[int]] = {}
    use_key = group_key.strip().lower() != "none"
    for idx, atoms in enumerate(atoms_list):
        raw_key = atoms.info.get(group_key) if use_key else None
        key = str(raw_key) if raw_key is not None else f"__idx_{idx}"
        groups.setdefault(key, []).append(idx)
    group_keys = list(groups.keys())
    return group_keys, [groups[key] for key in group_keys]


def _select_train_cover_groups(group_elements: Sequence[set[int]]) -> List[int]:
    if not group_elements:
        return []

    all_elements = set().union(*group_elements)
    elem_to_groups = {
        element: [i for i, elems in enumerate(group_elements) if element in elems]
        for element in sorted(all_elements)
    }

    selected: List[int] = []
    covered: set[int] = set()
    uncovered = set(all_elements)

    while uncovered:
        target = min(uncovered, key=lambda z: (len(elem_to_groups[z]), z))
        candidates = [i for i in elem_to_groups[target] if i not in selected]
        if not candidates:
            uncovered.remove(target)
            continue
        best = max(
            candidates,
            key=lambda i: (len(group_elements[i] & uncovered), len(group_elements[i]), -i),
        )
        selected.append(best)
        covered |= group_elements[best]
        uncovered = all_elements - covered

    return selected


def _select_diverse_indices(features: np.ndarray, k: int, seed: int) -> List[int]:
    n_items = len(features)
    if k <= 0 or n_items == 0:
        return []
    if k >= n_items:
        return list(range(n_items))

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n_items))]
    min_dist = np.sum((features - features[selected[0]]) ** 2, axis=1)
    min_dist[selected[0]] = -np.inf

    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        dist = np.sum((features - features[nxt]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf

    return selected


def _split_atoms_diversely(
    atoms_list: Sequence[Any],
    ratios: Sequence[float],
    group_key: str,
    seed: int,
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    ratios = _normalize_split_ratios(ratios)
    group_keys, grouped_indices = _group_indices(atoms_list, group_key=group_key)
    n_groups = len(grouped_indices)
    if n_groups < max(3, int(np.count_nonzero(ratios > 0))):
        raise ValueError("Need at least three groups to write train/valid/test splits")

    atom_features = _atoms_feature_matrix(atoms_list)
    group_features = np.vstack([atom_features[idxs].mean(axis=0) for idxs in grouped_indices])
    group_elements = [
        {int(z) for idx in idxs for z in np.asarray(atoms_list[idx].numbers, dtype=int).tolist()}
        for idxs in grouped_indices
    ]

    required_train = _select_train_cover_groups(group_elements)
    base_train, _, _ = _split_counts(n_groups, ratios)
    train_group_count = max(base_train, len(required_train))
    holdout_count = max(0, n_groups - train_group_count)
    valid_group_count, test_group_count = _allocate_counts(holdout_count, ratios[1:])
    train_group_count = n_groups - valid_group_count - test_group_count

    required_train_set = set(required_train)
    available = [i for i in range(n_groups) if i not in required_train_set]

    test_local = _select_diverse_indices(group_features[available], test_group_count, seed + 17)
    test_groups = [available[i] for i in test_local]

    remaining_for_valid = [i for i in available if i not in set(test_groups)]
    valid_local = _select_diverse_indices(group_features[remaining_for_valid], valid_group_count, seed + 31)
    valid_groups = [remaining_for_valid[i] for i in valid_local]

    split_group_sets = {
        "test": set(test_groups),
        "valid": set(valid_groups),
    }
    split_group_sets["train"] = {
        i for i in range(n_groups) if i not in split_group_sets["test"] and i not in split_group_sets["valid"]
    }

    frame_to_group = {}
    for group_idx, idxs in enumerate(grouped_indices):
        for idx in idxs:
            frame_to_group[idx] = group_idx

    split_atoms = {
        split: [atoms_list[i] for i in range(len(atoms_list)) if frame_to_group[i] in groups]
        for split, groups in split_group_sets.items()
    }

    summary = {
        "group_key": group_key,
        "n_frames": len(atoms_list),
        "n_groups": n_groups,
        "required_train_groups": [group_keys[i] for i in required_train],
        "splits": {},
    }
    for split, groups in split_group_sets.items():
        elems = sorted({int(z) for group_idx in groups for z in group_elements[group_idx]})
        summary["splits"][split] = {
            "n_frames": len(split_atoms[split]),
            "n_groups": len(groups),
            "elements": elems,
        }

    return split_atoms, summary


def _resolve_split_prefix(out_path: Path, split_prefix: str) -> Path:
    if split_prefix.strip():
        prefix = Path(split_prefix)
        return prefix if prefix.is_absolute() else out_path.parent / prefix
    return out_path.with_suffix("")


def _resolve_optional_output_path(base_path: Path, requested: str, suffix: str) -> Path:
    if requested.strip():
        path = Path(requested)
        return path if path.is_absolute() else base_path.parent / path
    ext = base_path.suffix or ".extxyz"
    return base_path.with_name(f"{base_path.stem}{suffix}{ext}")


def _write_split_files(
    atoms_list: Sequence[Any],
    out_path: Path,
    split_prefix: str,
    split_ext: str,
    ratios: Sequence[float],
    group_key: str,
    seed: int,
) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    split_atoms, summary = _split_atoms_diversely(
        atoms_list=atoms_list,
        ratios=ratios,
        group_key=group_key,
        seed=seed,
    )
    prefix = _resolve_split_prefix(out_path, split_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ext = split_ext.lstrip(".") or "xyz"

    split_paths: Dict[str, Path] = {}
    for split, images in split_atoms.items():
        path = prefix.parent / f"{prefix.name}-{split}.{ext}"
        if images:
            ase_write(path, images, format="extxyz")
            split_paths[split] = path

    return split_paths, summary


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("MP_API_KEY", ""), help="Materials Project API key (or set MP_API_KEY).")
    ap.add_argument("--out", default="MP-Dielectrics.extxyz", help="Output extxyz file.")
    ap.add_argument("--max-materials", type=int, default=0, help="Limit number of materials (0 = all).")
    ap.add_argument("--chunk-size", type=int, default=500, help="Chunk size for dielectric materials query.")
    ap.add_argument("--num-chunks", type=int, default=0, help="How many chunks to fetch (0 = all available).")
    ap.add_argument("--task-batch", type=int, default=100, help="Task doc batch size (<=100 recommended).")
    ap.add_argument(
        "--ase-rewrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read the written extxyz with ASE and rewrite it to normalize formatting (default: enabled).",
    )
    ap.add_argument(
        "--write-filtered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter the raw dataset and write a cleaned output file (default: disabled).",
    )
    ap.add_argument(
        "--filtered-out",
        default="",
        help="Optional path for the filtered output. Defaults to '<out>-filtered.extxyz'.",
    )
    ap.add_argument(
        "--filter-max-force-ev-a",
        type=float,
        default=0.05,
        help="Maximum allowed max force in eV/A for filtered structures (default: 0.05).",
    )
    ap.add_argument(
        "--filter-min-eps-eig",
        type=float,
        default=0.8,
        help="Minimum allowed dielectric eigenvalue for filtered structures (default: 0.8).",
    )
    ap.add_argument(
        "--filter-max-eps-scalar",
        type=float,
        default=200.0,
        help="Maximum allowed mean dielectric eigenvalue for filtered structures (default: 200).",
    )
    ap.add_argument(
        "--filter-max-asr-norm",
        type=float,
        default=0.3,
        help="Maximum allowed BEC acoustic sum rule Frobenius norm (default: 0.3).",
    )
    ap.add_argument(
        "--filter-max-stress-gpa",
        type=float,
        default=0.0,
        help="Optional maximum allowed stress magnitude in GPa for filtered structures (<=0 disables stress filtering).",
    )
    ap.add_argument(
        "--require-bec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require BEC labels in the filtered dataset (default: enabled).",
    )
    ap.add_argument(
        "--require-eps-total",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require total dielectric tensors in the filtered dataset (default: enabled).",
    )
    ap.add_argument(
        "--require-pol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require polarizability labels in the filtered dataset (default: enabled).",
    )
    ap.add_argument(
        "--require-stress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require stress labels in the filtered dataset (default: enabled).",
    )
    ap.add_argument(
        "--dedupe-filtered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deduplicate filtered structures with pymatgen StructureMatcher (default: disabled).",
    )
    ap.add_argument("--dedupe-ltol", type=float, default=0.2, help="StructureMatcher lattice tolerance for filtered deduplication.")
    ap.add_argument("--dedupe-stol", type=float, default=0.3, help="StructureMatcher site tolerance for filtered deduplication.")
    ap.add_argument("--dedupe-angle-tol", type=float, default=5.0, help="StructureMatcher angle tolerance in degrees for filtered deduplication.")
    ap.add_argument(
        "--write-splits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write train/valid/test split files. When filtered output is enabled, splits are written from the filtered dataset.",
    )
    ap.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALID", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Train/valid/test split ratios (default: 0.8 0.1 0.1).",
    )
    ap.add_argument(
        "--split-prefix",
        default="",
        help="Prefix for split output files. Defaults to the main output stem in the same directory.",
    )
    ap.add_argument(
        "--split-ext",
        default="xyz",
        help="Extension for split files (default: xyz). Files are written in extxyz format.",
    )
    ap.add_argument(
        "--split-group-key",
        default="material_id",
        help="Keep frames with the same atoms.info key together when splitting. Use 'none' to split per frame.",
    )
    ap.add_argument("--split-seed", type=int, default=0, help="Random seed for the diversity split.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set MP_API_KEY environment variable.")

    adaptor = AseAtomsAdaptor()

    written = 0
    skipped = 0
    fallback_source_used: Optional[str] = None

    with (
        MPRester(args.api_key) as mpr,
        MPRester(args.api_key, use_document_model=False) as mpr_raw_tasks,
        open(args.out, "w") as fh,
    ):
        # 1) Collect material_ids from dielectric endpoint (proxy for DFPT dielectric availability)
        if args.verbose:
            print("Fetching materials with dielectric data...")

        dielectric_num_chunks = args.num_chunks if args.num_chunks else None
        if dielectric_num_chunks is None and args.max_materials and args.max_materials > 0:
            dielectric_num_chunks = max(1, math.ceil(args.max_materials / args.chunk_size))

        diel_docs = mpr.materials.dielectric.search(
            fields=["material_id", "formula_pretty", "origins", "electronic", "ionic", "total"],
            chunk_size=args.chunk_size,
            num_chunks=dielectric_num_chunks,
        )

        dielectric_records: List[Tuple[str, Optional[str], Optional[str], Any]] = []
        material_ids = []
        for d in diel_docs:
            material_id = _safe_get(d, ["material_id"], default=None)
            if material_id is not None:
                mid = str(material_id)
                material_ids.append(mid)
                dielectric_records.append(
                    (
                        mid,
                        _safe_get(d, ["formula_pretty"], default=None),
                        _pick_origin_task_id(_safe_get(d, ["origins"], default=None), "dielectric"),
                        d,
                    )
                )
        if args.max_materials and args.max_materials > 0:
            material_ids = material_ids[: args.max_materials]
            dielectric_records = dielectric_records[: args.max_materials]

        if args.verbose:
            print(f"Found {len(material_ids)} materials with dielectric data.")

        # 2) Normalize dielectric provenance task IDs from the dielectric docs themselves
        mid_to_meta: List[Tuple[str, Optional[str], Optional[str], Any]] = []
        for mid, formula_pretty, diel_tid, diel_doc in dielectric_records:
            mid_to_meta.append((mid, formula_pretty, _canonical_task_id(diel_tid), diel_doc))

        # 3) Fetch dielectric task docs in batches
        dielectric_task_ids = [t[2] for t in mid_to_meta if t[2]]

        diel_tasks: Dict[str, Any] = {}

        diel_batches = list(_batched([x for x in dielectric_task_ids if x], args.task_batch))

        if tqdm is not None:
            diel_batches = tqdm(diel_batches, desc="Fetch DFPT tasks", unit="batch")
        for tids in diel_batches:
            try:
                tdocs = mpr_raw_tasks.materials.tasks.search(task_ids=tids, all_fields=True)
            except Exception as e:
                if args.verbose:
                    print(f"DFPT task batch failed: {e}")
                continue
            for td in tdocs:
                task_id = _canonical_task_id(_safe_get(td, ["task_id"], default=None))
                if task_id is not None:
                    diel_tasks[task_id] = td

        # The current API can fail direct legacy DFPT task lookups by task_id.
        # Recover missing task docs by querying tasks with formula filters and
        # then selecting the specific provenance task IDs locally.
        missing_task_ids = {tid for tid in dielectric_task_ids if tid and tid not in diel_tasks}
        if missing_task_ids:
            formula_to_needed_ids: Dict[str, set[str]] = {}
            for _, formula_pretty, diel_tid, _ in mid_to_meta:
                if formula_pretty and diel_tid in missing_task_ids:
                    formula_to_needed_ids.setdefault(str(formula_pretty), set()).add(diel_tid)

            formula_iter: Iterable[str] = sorted(formula_to_needed_ids)
            if tqdm is not None:
                formula_iter = tqdm(formula_iter, desc="Recover DFPT tasks by formula", unit="formula")

            for formula in formula_iter:
                try:
                    tdocs = mpr.materials.tasks.search(formula=formula, all_fields=True)
                except Exception as e:
                    if args.verbose:
                        print(f"Formula fallback failed for {formula}: {e}")
                    continue

                wanted_ids = formula_to_needed_ids.get(formula, set())

                for td in tdocs:
                    task_id = _canonical_task_id(_safe_get(td, ["task_id"], default=None))
                    if task_id in wanted_ids:
                        diel_tasks[task_id] = td

        # 4) Join per material and write frames
        join_iter = mid_to_meta
        if tqdm is not None:
            join_iter = tqdm(join_iter, desc="Write extxyz", unit="material")

        for mid, _, diel_tid, diel_doc in join_iter:
            if not diel_tid:
                skipped += 1
                continue

            dfpt = diel_tasks.get(diel_tid)

            eps_static, eps_ionic, eps_total = _extract_dielectric_doc_tensors(diel_doc)
            born = None
            if dfpt is not None:
                task_eps_static, task_eps_ionic, task_eps_total, born = _extract_dfpt_tensors(dfpt)
                if eps_static is None:
                    eps_static = task_eps_static
                if eps_ionic is None:
                    eps_ionic = task_eps_ionic
                if eps_total is None:
                    eps_total = task_eps_total
                structure, energy, forces, stress = _extract_task_quantities(dfpt)
            else:
                structure, energy, forces, stress = None, None, None, None

            if structure is None:
                skipped += 1
                continue

            atoms = adaptor.get_atoms(structure)

            arrays: Dict[str, np.ndarray] = {}

            info: Dict[str, Any] = {
                "material_id": mid,
                "dielectric_task_id": diel_tid,
                "REF_energy": energy,
                "REF_stress_units": "eV/Angstrom^3",
            }

            # forces should be (N,3)
            if forces is not None and forces.shape == (len(atoms), 3):
                arrays["REF_forces"] = forces
            else:
                info["config_forces_weight"] = 0.0

            # BECs should be (N,3,3) -> store as 9 per-atom
            if born is not None and born.ndim == 3 and born.shape[0] == len(atoms) and born.shape[1:] == (3, 3):
                arrays["REF_becs"] = born.reshape(len(atoms), 9)
            else:
                info["config_becs_weight"] = 0.0

            # Stress: convert to eV/Å^3 then flatten
            stress_ev_a3 = _stress_to_ev_a3(stress)
            sflat = _stress_to_flat(stress_ev_a3)
            if sflat is not None:
                info["REF_stress"] = sflat
            else:
                info["config_stress_weight"] = 0.0

            # Dielectric tensors
            if eps_static is not None and eps_static.shape == (3, 3):
                info["REF_epsilon_static"] = eps_static.reshape(-1)
                # SI susceptibility tensor (dimensionless): chi = eps_r - I
                info["REF_polarizability"] = (eps_static - np.eye(3)).reshape(-1)
            else:
                info["config_polarizability_weight"] = 0.0

            if eps_ionic is not None and eps_ionic.shape == (3, 3):
                info["REF_epsilon_ionic"] = eps_ionic.reshape(-1)

            if eps_total is not None and eps_total.shape == (3, 3):
                info["REF_epsilon_total"] = eps_total.reshape(-1)

            _write_extxyz_frame(fh, atoms, info, arrays)
            written += 1

    out_path = Path(args.out)
    if written == 0:
        fallback_atoms, fallback_source = _load_local_dielectric_fallback(
            prefer_filtered=args.write_filtered or args.write_splits
        )
        if fallback_atoms is None:
            raise RuntimeError(
                "MP API returned 0 writable dielectric frames and no local fallback dataset was found."
            )
        ase_write(out_path, fallback_atoms, format="extxyz")
        written = len(fallback_atoms)
        fallback_source_used = fallback_source
        print(
            f"Used local dielectric fallback from {fallback_source} because "
            "the current MP tasks endpoint returned 0 writable task documents."
        )

    postprocess_atoms: Optional[List[Any]] = None
    need_raw_atoms = args.ase_rewrite or args.write_filtered or args.write_splits
    if need_raw_atoms:
        postprocess_atoms = _read_atoms_list(out_path)

    if args.ase_rewrite:
        postprocess_atoms = _rewrite_extxyz_with_ase(out_path, atoms_list=postprocess_atoms)
        if args.verbose:
            print(f"ASE rewrite completed for {out_path}")

    split_source_atoms = postprocess_atoms
    split_source_path = out_path

    filtered_path: Optional[Path] = None
    filtered_atoms: Optional[List[Any]] = None
    filter_summary: Optional[Dict[str, Any]] = None
    dedupe_summary: Optional[Dict[str, Any]] = None
    if args.write_filtered:
        if postprocess_atoms is None:
            postprocess_atoms = _read_atoms_list(out_path)

        max_stress_ev_a3 = None
        if args.filter_max_stress_gpa > 0:
            max_stress_ev_a3 = args.filter_max_stress_gpa / GPA_PER_EV_A3

        filtered_atoms, filter_summary = _filter_atoms(
            postprocess_atoms,
            max_force_ev_a=args.filter_max_force_ev_a,
            min_eps_eig=args.filter_min_eps_eig,
            max_eps_scalar=args.filter_max_eps_scalar,
            max_asr_norm=args.filter_max_asr_norm,
            max_stress_ev_a3=max_stress_ev_a3,
            require_bec=args.require_bec,
            require_eps_total=args.require_eps_total,
            require_pol=args.require_pol,
            require_stress=args.require_stress,
        )

        if args.dedupe_filtered:
            filtered_atoms, dedupe_summary = _dedupe_atoms(
                filtered_atoms,
                ltol=args.dedupe_ltol,
                stol=args.dedupe_stol,
                angle_tol=args.dedupe_angle_tol,
            )

        filtered_path = _resolve_optional_output_path(out_path, args.filtered_out, "-filtered")
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(filtered_path, filtered_atoms, format="extxyz")
        if args.ase_rewrite:
            filtered_atoms = _rewrite_extxyz_with_ase(filtered_path, atoms_list=filtered_atoms)
        split_source_atoms = filtered_atoms
        split_source_path = filtered_path

    split_paths: Dict[str, Path] = {}
    split_summary: Optional[Dict[str, Any]] = None
    if args.write_splits:
        if split_source_atoms is None:
            split_source_atoms = _read_atoms_list(split_source_path)
        split_paths, split_summary = _write_split_files(
            atoms_list=split_source_atoms,
            out_path=split_source_path,
            split_prefix=args.split_prefix,
            split_ext=args.split_ext,
            ratios=args.split_ratios,
            group_key=args.split_group_key,
            seed=args.split_seed,
        )

    if fallback_source_used is not None:
        print(f"Done. Restored {written} frames from local fallback to {args.out}.")
    else:
        print(f"Done. Wrote {written} raw frames to {args.out}. Skipped {skipped} materials.")
    if filtered_path is not None and filter_summary is not None:
        print(
            "Filtered dataset:"
            f" kept {filter_summary['n_kept']} / {filter_summary['n_input']}"
            f" frames -> {filtered_path}"
        )
        if filter_summary["drop_reason_counts"]:
            print("  Drop reasons:")
            for reason, count in sorted(
                filter_summary["drop_reason_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                print(f"    {reason}: {count}")
        if dedupe_summary is not None:
            print(
                "  Deduplication:"
                f" kept {dedupe_summary['n_kept']} / {dedupe_summary['n_input']}"
                f" structures across {dedupe_summary['n_groups']} matcher groups"
            )
    if split_paths:
        for split in ("train", "valid", "test"):
            path = split_paths.get(split)
            if path is None:
                continue
            n_frames = split_summary["splits"][split]["n_frames"] if split_summary is not None else "?"
            n_groups = split_summary["splits"][split]["n_groups"] if split_summary is not None else "?"
            print(f"  {split:>5}: {n_frames} frames across {n_groups} groups -> {path}")


if __name__ == "__main__":
    main()
