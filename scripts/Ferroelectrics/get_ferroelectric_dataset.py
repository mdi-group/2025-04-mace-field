#!/usr/bin/env python3
"""
Fetch the MPContribs ferroelectric dataset, write the combined extxyz,
and optionally produce train/valid/test splits plus isolated atoms.

This is a script version of get_ferroelectric_dataset.ipynb with the same
core behavior and output conventions.

Requirements:
  pip install mpcontribs-client pymatgen ase numpy pandas requests scikit-learn tqdm

Usage:
  export MP_API_KEY="YOUR_KEY"
  python get_ferroelectric_dataset.py --out ferroelectric.xyz
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from ase import Atoms
from ase.data import chemical_symbols, chemical_symbols as ASE_SYMBOLS
from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from mpcontribs.client import Client
except Exception as exc:  # pragma: no cover - import guard
    Client = None
    _mpcontribs_error = exc

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception as exc:  # pragma: no cover - import guard
    StratifiedGroupKFold = None
    _sklearn_error = exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


FERROELECTRIC_polarizATION_SCALE = 1602.17663
DEFAULT_ELECTRIC_FIELD = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_EXT_SPONTANEOUS_SCALE = 1602.17663
GPA_PER_EV_A3 = 160.21766208
# MPContribs ferroelectric workflow payloads expose VASP-style raw stress values.
# Those follow the usual VASP sign convention and are reported in kbar-like units,
# so convert with the sign flip required to match ASE/MACE stress in eV/Angstrom^3.
EV_A3_PER_KBAR = -0.1 / GPA_PER_EV_A3

HARDCODED_ISOLATED_E0S: "OrderedDict[int, float]" = OrderedDict(
    [
        (1, -3.667168021358939),
        (2, -1.3320953124042916),
        (3, -3.482100566595956),
        (4, -4.736697230897597),
        (5, -7.724935420523256),
        (6, -8.405573550273285),
        (7, -7.360100452662763),
        (8, -7.28459863421322),
        (9, -4.896490881731322),
        (10, 1.3917755836700962e-12),
        (11, -2.7593613569762425),
        (12, -2.814047612069227),
        (13, -4.846881245288104),
        (14, -7.694793133351899),
        (15, -6.9632957911820235),
        (16, -4.672630400190884),
        (17, -2.8116892814008096),
        (18, -0.06259504416367478),
        (19, -2.6176454856894793),
        (20, -5.390461060484104),
        (21, -7.8857952163517675),
        (22, -10.268392986214433),
        (23, -8.665147785496703),
        (24, -9.233050763772013),
        (25, -8.304951520770791),
        (26, -7.0489865771593765),
        (27, -5.577439766222147),
        (28, -5.172747618813715),
        (29, -3.2520726958619472),
        (30, -1.2901611618726314),
        (31, -3.527082192997912),
        (32, -4.70845955030298),
        (33, -3.9765109025623238),
        (34, -3.886231055836541),
        (35, -2.5184940099633986),
        (36, 6.766947645687137),
        (37, -2.5634958965928316),
        (38, -4.938005211501922),
        (39, -10.149818838085771),
        (40, -11.846857579882572),
        (41, -12.138896361658485),
        (42, -8.791678800595722),
        (43, -8.78694939675911),
        (44, -7.78093221529871),
        (45, -6.850021409115055),
        (46, -4.891019073240479),
        (47, -2.0634296773864045),
        (48, -0.6395695518943755),
        (49, -2.7887442084286693),
        (50, -3.818604275441892),
        (51, -3.587068329278862),
        (52, -2.8804045971118897),
        (53, -1.6355986842433357),
        (54, 9.846723842807721),
        (55, -2.765284507132287),
        (56, -4.990956432167774),
        (57, -8.933684809576345),
        (58, -8.735591176647514),
        (59, -8.018966025544966),
        (60, -8.251491970213372),
        (61, -7.591719594359237),
        (62, -8.169659881166858),
        (63, -13.592664636171698),
        (64, -18.517523458456985),
        (65, -7.647396572993602),
        (66, -8.122981037851925),
        (67, -7.607787319678067),
        (68, -6.85029094445494),
        (69, -7.8268821327130365),
        (70, -3.584786591677161),
        (71, -7.455406192077973),
        (72, -12.796283502572146),
        (73, -14.108127281277586),
        (74, -9.354916969477486),
        (75, -11.387537567890853),
        (76, -9.621909492152557),
        (77, -7.324393429417677),
        (78, -5.3046964808341945),
        (79, -2.380092582080244),
        (80, 0.24948924158195362),
        (81, -2.3239789120665026),
        (82, -3.730042357127322),
        (83, -3.438792347649683),
        (89, -5.062878214511315),
        (90, -11.02462566385297),
        (91, -12.265613551943261),
        (92, -13.855648206100362),
        (93, -14.933092020258243),
        (94, -15.282826131998245),
    ]
)


def z_tuple(at: Atoms) -> Tuple[int, ...]:
    """Sorted unique atomic numbers in a structure (stable for grouping)."""
    return tuple(sorted(set(int(z) for z in at.get_atomic_numbers())))


def z_union(atoms_list: List[Atoms]) -> List[int]:
    """Sorted unique Z across a list of structures."""
    return sorted({int(z) for at in atoms_list for z in at.get_atomic_numbers()})


def symbols_from_z(z_list: List[int]) -> List[str]:
    return [ASE_SYMBOLS[z] for z in z_list]


def get_ref_polarization_vec(at: Atoms, key: str = "REF_polarization") -> np.ndarray:
    """Return polarization vector (length-3) as np.ndarray[float64]."""
    if key not in at.info:
        raise KeyError(f"atoms.info['{key}'] missing")
    arr = np.asarray(at.info[key], dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"atoms.info['{key}'] must be length-3; got shape {arr.shape}")
    return arr


def p_norm(arr3: np.ndarray) -> float:
    return float(np.linalg.norm(arr3))


def p_octant(arr3: np.ndarray) -> int:
    """Encode the polarization octant as an integer in [0..7]."""
    sx = 1 if arr3[0] < 0 else 0
    sy = 1 if arr3[1] < 0 else 0
    sz = 1 if arr3[2] < 0 else 0
    return (sx << 0) | (sy << 1) | (sz << 2)


def get_ref_energy(
    at: Atoms,
    keys: Tuple[str, ...] = ("REF_energy", "energy", "E", "dft_energy", "total_energy"),
) -> Optional[float]:
    for key in keys:
        if key in at.info:
            try:
                return float(at.info[key])
            except Exception:
                pass
    try:
        return float(at.get_potential_energy())
    except Exception:
        return None


def counts_vector_by_z(at: Atoms, z_order: List[int]) -> np.ndarray:
    """Element count vector aligned to z_order."""
    counts = {z: 0 for z in z_order}
    for z in at.get_atomic_numbers():
        counts[int(z)] += 1
    return np.array([counts[z] for z in z_order], dtype=float)


def fit_e0s(
    atoms_list: List[Atoms],
    z_all: List[int],
    ridge_lambda: float = 0.0,
) -> "OrderedDict[int, float]":
    """Least-squares isolated-atom reference energies fitted over all structures."""
    energy_rows = []
    for at in atoms_list:
        energy = get_ref_energy(at)
        if energy is not None:
            energy_rows.append((counts_vector_by_z(at, z_all), energy))

    if len(energy_rows) == 0:
        warnings.warn("No reference energies found; returning E0=0.0 for all Z.")
        e0_vec = np.zeros(len(z_all), dtype=float)
    else:
        a_mat = np.stack([row[0] for row in energy_rows], axis=0)
        b_vec = np.array([row[1] for row in energy_rows], dtype=float)
        if ridge_lambda > 0.0:
            ata = a_mat.T @ a_mat
            atb = a_mat.T @ b_vec
            e0_vec = np.linalg.solve(ata + ridge_lambda * np.eye(ata.shape[0]), atb)
        else:
            e0_vec, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)

    return OrderedDict((int(z), float(e0)) for z, e0 in zip(z_all, e0_vec))


def make_mace_splits(
    atoms_list: List[Atoms],
    *,
    n_splits: int = 10,
    test_fold: int = 0,
    val_fold: int = 1,
    pol_key: str = "REF_polarization",
    random_state: int = 42,
    strat_bins: int = 10,
    use_octant: bool = True,
    min_stratum_size: int = 2,
    ridge_lambda: float = 0.0,
    allow_branch_cross_split: bool = True,
    min_total_for_eval: int = 3,
    require_same_eval_set: bool = True,
) -> Tuple[
    Dict[str, List[Atoms]],
    Dict[str, List[int]],
    Dict[str, List[str]],
    "OrderedDict[int, float]",
    List[int],
    List[str],
]:
    """
    Grouped+stratified split with post-fixes for element coverage.

    If allow_branch_cross_split=True, each sample is its own group for the splitter,
    so frames from the same material/branch can be distributed across splits.
    If False, we group by chemistry (z_tuple), keeping them together.
    """
    if len(atoms_list) < 3:
        raise ValueError("Need at least 3 structures for train/valid/test splits.")
    if StratifiedGroupKFold is None:
        raise RuntimeError(
            "scikit-learn is required for grouped+stratified split. "
            f"Import error: {_sklearn_error}"
        )

    records = []
    for idx, at in enumerate(atoms_list):
        pol = get_ref_polarization_vec(at, pol_key)
        records.append(
            {
                "idx": idx,
                "group": z_tuple(at),
                "P_x": pol[0],
                "P_y": pol[1],
                "P_z": pol[2],
                "P_norm": p_norm(pol),
                "P_oct": p_octant(pol),
            }
        )
    df = pd.DataFrame(records)

    z_all = z_union(atoms_list)
    symbols_all = symbols_from_z(z_all)

    try:
        df["mag_bin"] = pd.qcut(df["P_norm"], q=strat_bins, labels=False, duplicates="drop")
    except Exception:
        df["mag_bin"] = pd.cut(df["P_norm"], bins=strat_bins, labels=False, include_lowest=True)

    if use_octant:
        y_code = df["mag_bin"].astype(int) * 8 + df["P_oct"].astype(int)
        counts = Counter(y_code.tolist())
        if any(count < min_stratum_size for count in counts.values()):
            y_strata = df["mag_bin"].astype(int).to_numpy()
        else:
            y_strata = y_code.to_numpy()
    else:
        y_strata = df["mag_bin"].astype(int).to_numpy()

    if allow_branch_cross_split:
        groups_for_split = df["idx"].to_numpy()
    else:
        groups_for_split = df["group"]

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X=df.index, y=y_strata, groups=groups_for_split))
    test_idx = np.array(folds[test_fold][1], dtype=int)
    val_idx = np.array(folds[val_fold][1], dtype=int)
    holdout = np.union1d(test_idx, val_idx)
    all_idx = df["idx"].to_numpy(dtype=int)
    train_idx = np.setdiff1d(all_idx, holdout)

    elems_per_idx: List[set[int]] = [
        set(int(z) for z in at.get_atomic_numbers()) for at in atoms_list
    ]

    total_count = Counter()
    for elems in elems_per_idx:
        for z in elems:
            total_count[z] += 1

    rare_elems = {z for z, count in total_count.items() if count < min_total_for_eval}

    split_of = {i: "train" for i in train_idx}
    split_of.update({i: "valid" for i in val_idx})
    split_of.update({i: "test" for i in test_idx})

    def rebuild_lists() -> Tuple[List[int], List[int], List[int]]:
        train = [i for i, split in split_of.items() if split == "train"]
        valid = [i for i, split in split_of.items() if split == "valid"]
        test = [i for i, split in split_of.items() if split == "test"]
        return train, valid, test

    for idx, split in list(split_of.items()):
        if split in ("valid", "test") and (elems_per_idx[idx] & rare_elems):
            split_of[idx] = "train"
    train_idx, val_idx, test_idx = rebuild_lists()

    def elems_in(indices: List[int]) -> set[int]:
        elems: set[int] = set()
        for idx in indices:
            elems |= elems_per_idx[idx]
        return {z for z in elems if z not in rare_elems}

    target_eval_elems = {z for z in z_all if z not in rare_elems}

    for z in list(target_eval_elems):
        if not any(z in elems_per_idx[idx] for idx in train_idx):
            donors = [idx for idx in (val_idx + test_idx) if z in elems_per_idx[idx]]
            if donors:
                picked = min(donors, key=lambda idx: len(elems_per_idx[idx]))
                split_of[picked] = "train"
                train_idx, val_idx, test_idx = rebuild_lists()

    if require_same_eval_set and target_eval_elems:

        def move_one_for(z: int, dest: str) -> bool:
            candidates = [
                idx
                for idx, split in split_of.items()
                if split != dest and (z in elems_per_idx[idx]) and not (elems_per_idx[idx] & rare_elems)
            ]
            if not candidates:
                return False

            def score(idx: int) -> Tuple[int, int]:
                src = split_of[idx]
                src_list = train_idx if src == "train" else (val_idx if src == "valid" else test_idx)
                src_count_z = sum(1 for j in src_list if z in elems_per_idx[j])
                return (-src_count_z, len(elems_per_idx[idx]))

            picked = sorted(candidates, key=score)[0]
            split_of[picked] = dest
            return True

        changed = True
        iterations = 1000
        while changed and iterations > 0:
            changed = False
            iterations -= 1
            train_idx, val_idx, test_idx = rebuild_lists()
            sets_now = {
                "train": elems_in(train_idx),
                "valid": elems_in(val_idx),
                "test": elems_in(test_idx),
            }
            for dest in ("valid", "test"):
                for z in list(target_eval_elems - sets_now[dest]):
                    if move_one_for(z, dest):
                        changed = True

            train_idx, val_idx, test_idx = rebuild_lists()
            sets_now = {
                "train": elems_in(train_idx),
                "valid": elems_in(val_idx),
                "test": elems_in(test_idx),
            }
            for z in list(target_eval_elems - sets_now["train"]):
                if move_one_for(z, "train"):
                    changed = True

        train_idx, val_idx, test_idx = rebuild_lists()
        sets_now = {
            "train": elems_in(train_idx),
            "valid": elems_in(val_idx),
            "test": elems_in(test_idx),
        }
        common = sets_now["train"] & sets_now["valid"] & sets_now["test"]
        if common != target_eval_elems:
            missing_report = {key: sorted(target_eval_elems - val) for key, val in sets_now.items()}
            warnings.warn(
                "Could not make all splits share the exact same evaluation elements.\n"
                f"Target (non-rare) = {sorted(target_eval_elems)}\n"
                f"Missing per split: {missing_report}"
            )

    splits = {
        "train": [atoms_list[idx] for idx in train_idx],
        "valid": [atoms_list[idx] for idx in val_idx],
        "test": [atoms_list[idx] for idx in test_idx],
    }

    split_elements_z = {key: z_union(val) for key, val in splits.items()}
    split_elements_symbols = {
        key: symbols_from_z(zs) for key, zs in split_elements_z.items()
    }
    e0_z = fit_e0s(atoms_list, z_all, ridge_lambda=ridge_lambda)

    return splits, split_elements_z, split_elements_symbols, e0_z, z_all, symbols_all


def create_isolated_atoms(
    e0s: Mapping[int, float],
    box: float = 20.0,
    energy_key: str = "energy",
    config_type: str = "IsolatedAtom",
) -> List[Atoms]:
    """Build single-atom ASE Atoms objects for each element in e0s."""
    items = e0s.items() if isinstance(e0s, OrderedDict) else sorted(e0s.items(), key=lambda kv: kv[0])
    atoms_list: List[Atoms] = []
    for z, e0 in items:
        if not (0 < z < len(chemical_symbols)):
            raise ValueError(f"Atomic number {z} is out of range for ASE chemical symbols.")
        sym = chemical_symbols[z]
        atom = Atoms(
            symbols=[sym],
            positions=[[0.0, 0.0, 0.0]],
            cell=[box, box, box],
            pbc=[False, False, False],
        )
        atom.info["config_type"] = config_type
        atom.info["Z"] = int(z)
        atom.info["symbol"] = sym
        atom.info[energy_key] = float(e0)
        atom.info["name"] = f"{config_type}-{z}-{sym}"
        atoms_list.append(atom)
    return atoms_list


def _resolve_split_prefix(out_path: Path, split_prefix: str) -> Path:
    if split_prefix.strip():
        prefix = Path(split_prefix)
        return prefix if prefix.is_absolute() else out_path.parent / prefix
    return out_path.with_suffix("")


def _require_mpcontribs() -> None:
    if Client is None:
        raise RuntimeError(
            "mpcontribs-client is required to fetch ferroelectric data. "
            f"Import error: {_mpcontribs_error}"
        )


def _download_json_gz(session: requests.Session, url: str, timeout: float) -> dict:
    response = session.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as fh:
        payload = json.loads(fh.read())
    return payload


def fetch_ferroelectric_atoms(
    api_key: str,
    *,
    project: str = "ferroelectrics",
    workflow_status: str = "COMPLETED",
    max_contributions: int = 0,
    timeout: float = 60.0,
    electric_field: np.ndarray = DEFAULT_ELECTRIC_FIELD,
    verbose: bool = False,
) -> List[Atoms]:
    """Fetch the MPContribs ferroelectric dataset used in the notebook."""
    _require_mpcontribs()

    client = Client(apikey=api_key, project=project)
    query = {"data__workflow__status__in": workflow_status}
    contributions = client.query_contributions(
        query=query,
        fields=["attachments"],
        paginate=True,
    )["data"]
    if max_contributions > 0:
        contributions = contributions[:max_contributions]

    session = requests.Session()
    adaptor = AseAtomsAdaptor()
    atoms_list: List[Atoms] = []

    iterator = contributions
    if tqdm is not None:
        iterator = tqdm(contributions, desc="Fetch ferroelectrics", unit="contrib")

    for entry in iterator:
        attachments = entry.get("attachments") or []
        if not attachments:
            continue

        component_id = attachments[-1]["id"]
        url = f"https://contribs.materialsproject.org/contributions/component/{component_id}"
        payload = _download_json_gz(session, url, timeout=timeout)

        structures = payload.get("structures", [])
        same_branch = payload.get("same_branch_polarization", [])
        energies = payload.get("energies", [])
        stresses = payload.get("stresses", [])
        forces = payload.get("forces", [])

        for idx, structure in enumerate(structures):
            atom = adaptor.get_atoms(Structure.from_dict(structure), msonable=False)

            if idx < len(same_branch) and same_branch[idx] is not None:
                atom.info["nonpolar_mpid"] = payload.get("nonpolar_id")
                atom.info["polar_mpid"] = payload.get("polar_id")
                atom.info["REF_energy"] = float(np.asarray(energies[idx], dtype=float))
                # MPContribs ferroelectric stresses come from atomate/VASP static
                # calculations and follow the raw VASP stress sign convention. MACE
                # expects ASE-style stress in eV/Angstrom^3, so convert
                # kbar -> GPa -> eV/Angstrom^3 with the required sign flip.
                atom.info["REF_stress"] = (
                    np.asarray(stresses[idx], dtype=float).reshape(-1) * EV_A3_PER_KBAR
                )
                atom.info["REF_polarization"] = (
                    np.asarray(same_branch[idx], dtype=float).reshape(3)
                    / FERROELECTRIC_polarizATION_SCALE
                )
                atom.info["REF_electric_field"] = np.asarray(electric_field, dtype=float).reshape(3)

            if idx < len(forces):
                atom.arrays["REF_forces"] = np.asarray(forces[idx], dtype=float)

            atoms_list.append(atom)

    if verbose:
        print(f"Fetched {len(atoms_list)} ferroelectric structures from project '{project}'.")
    return atoms_list


def fetch_ferroelectric_ext_atoms(
    api_key: str,
    *,
    project: str = "ferroelectrics_ext",
    max_contributions: int = 0,
    timeout: float = 60.0,
    spontaneous_scale: float = DEFAULT_EXT_SPONTANEOUS_SCALE,
    verbose: bool = False,
) -> List[Atoms]:
    """Fetch the sidecar ferroelectrics_ext dataset from the notebook."""
    _require_mpcontribs()

    client = Client(apikey=api_key, project=project)
    contributions = client.query_contributions(
        fields=["identifier", "structures", "data.Polarization.value"],
        paginate=True,
    )["data"]
    if max_contributions > 0:
        contributions = contributions[:max_contributions]

    session = requests.Session()
    atoms_list: List[Atoms] = []

    iterator = contributions
    if tqdm is not None:
        iterator = tqdm(contributions, desc="Fetch ferroelectrics_ext", unit="contrib")

    for entry in iterator:
        structures = entry.get("structures") or []
        if not structures:
            continue

        for idx in (0, 2):
            if idx >= len(structures):
                continue

            component = structures[idx]
            component_id = component["id"]
            name = component["name"]
            url = f"https://contribs.materialsproject.org/contributions/component/{component_id}"

            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as fh:
                cif_data = fh.read().decode("utf-8")
            atom = ase_read(io.StringIO(cif_data), format="cif")

            atom.info["mpid"] = entry.get("identifier")
            atom.info["type"] = "polar" if idx == 0 else "nonpolar"

            pol_val = (
                entry.get("data", {})
                .get("Polarization", {})
                .get("value")
            )
            if pol_val is not None:
                atom.info["spontaneous_polarization"] = pol_val
                atom.info["spontaneous_polarization_scaled"] = float(pol_val) * spontaneous_scale

            atom.info["component_name"] = name
            atoms_list.append(atom)

    if verbose:
        print(f"Fetched {len(atoms_list)} structures from project '{project}'.")
    return atoms_list


def _write_atoms(path: Path, atoms_list: Sequence[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(path, list(atoms_list), format="extxyz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MP_API_KEY", ""),
        help="Materials Project / MPContribs API key (or set MP_API_KEY).",
    )
    parser.add_argument(
        "--project",
        default="ferroelectrics",
        help="MPContribs project name for the main ferroelectric dataset.",
    )
    parser.add_argument(
        "--workflow-status",
        default="COMPLETED",
        help="Workflow status filter for the main project query.",
    )
    parser.add_argument(
        "--max-contributions",
        type=int,
        default=0,
        help="Limit the number of queried contributions (0 = all).",
    )
    parser.add_argument(
        "--out",
        default="MP-Ferroelectrics.xyz",
        help="Output extxyz for the combined ferroelectric dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for MPContribs component downloads.",
    )
    parser.add_argument(
        "--electric-field",
        nargs=3,
        type=float,
        metavar=("EX", "EY", "EZ"),
        default=DEFAULT_ELECTRIC_FIELD.tolist(),
        help="Field label to attach to ferroelectric structures (default: 0.01 0.01 0.01).",
    )

    parser.add_argument(
        "--write-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write ferroelectric-train/valid/test split files (default: enabled).",
    )
    parser.add_argument(
        "--split-prefix",
        default="",
        help="Prefix for split files. Defaults to the main output stem in the same directory.",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Number of StratifiedGroupKFold folds.")
    parser.add_argument("--test-fold", type=int, default=0, help="Fold index to use as test split.")
    parser.add_argument("--val-fold", type=int, default=1, help="Fold index to use as validation split.")
    parser.add_argument("--pol-key", default="REF_polarization", help="polarization info key for splitting.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the splitter.")
    parser.add_argument("--strat-bins", type=int, default=10, help="Number of polarization-magnitude bins.")
    parser.add_argument(
        "--use-octant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include polarization octant in stratification (default: enabled).",
    )
    parser.add_argument(
        "--min-stratum-size",
        type=int,
        default=2,
        help="Fallback to magnitude-only strata when a combined stratum is smaller than this.",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=0.0,
        help="Ridge regularization for fitted isolated-atom energies.",
    )
    parser.add_argument(
        "--allow-branch-cross-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow frames from the same branch/material to land in different splits (default: enabled).",
    )
    parser.add_argument(
        "--min-total-for-eval",
        type=int,
        default=3,
        help="Elements appearing fewer times than this are forced into train only.",
    )
    parser.add_argument(
        "--require-same-eval-set",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to enforce identical non-rare element coverage across train/valid/test.",
    )

    parser.add_argument(
        "--write-isolated-atoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write isolated_atoms.xyz (default: enabled).",
    )
    parser.add_argument(
        "--isolated-out",
        default="isolated_atoms.xyz",
        help="Output path for isolated atoms.",
    )
    parser.add_argument(
        "--isolated-box",
        type=float,
        default=25.0,
        help="Vacuum box length in Angstrom for isolated atoms.",
    )
    parser.add_argument(
        "--isolated-energy-source",
        choices=("hardcoded", "fit"),
        default="hardcoded",
        help="Use the notebook's hardcoded isolated-atom energies or fit them from the dataset.",
    )

    parser.add_argument(
        "--write-ext",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also fetch and write the ferroelectrics_ext sidecar dataset.",
    )
    parser.add_argument(
        "--ext-project",
        default="ferroelectrics_ext",
        help="MPContribs project name for the optional ext dataset.",
    )
    parser.add_argument(
        "--max-ext-contributions",
        type=int,
        default=0,
        help="Limit queried contributions for the optional ext dataset (0 = all).",
    )
    parser.add_argument(
        "--ext-out",
        default="ferroelectric_ext.xyz",
        help="Output path for the optional ext dataset.",
    )
    parser.add_argument(
        "--ext-spontaneous-scale",
        type=float,
        default=DEFAULT_EXT_SPONTANEOUS_SCALE,
        help="Scale factor applied to the ext spontaneous polarization value.",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set MP_API_KEY.")

    out_path = Path(args.out)
    atoms_list = fetch_ferroelectric_atoms(
        args.api_key,
        project=args.project,
        workflow_status=args.workflow_status,
        max_contributions=args.max_contributions,
        timeout=args.timeout,
        electric_field=np.asarray(args.electric_field, dtype=float),
        verbose=args.verbose,
    )
    _write_atoms(out_path, atoms_list)

    split_summary = None
    split_paths: Dict[str, Path] = {}
    e0_z: "OrderedDict[int, float]" = fit_e0s(atoms_list, z_union(atoms_list), ridge_lambda=args.ridge_lambda)

    if args.write_splits:
        split_candidates = [at for at in atoms_list if args.pol_key in at.info]
        skipped_for_split = len(atoms_list) - len(split_candidates)
        if skipped_for_split > 0:
            warnings.warn(
                f"Skipping {skipped_for_split} structures without {args.pol_key} when building splits."
            )

        splits, split_z, split_syms, e0_z, z_all, syms_all = make_mace_splits(
            split_candidates,
            n_splits=args.n_splits,
            test_fold=args.test_fold,
            val_fold=args.val_fold,
            pol_key=args.pol_key,
            random_state=args.random_state,
            strat_bins=args.strat_bins,
            use_octant=args.use_octant,
            min_stratum_size=args.min_stratum_size,
            ridge_lambda=args.ridge_lambda,
            allow_branch_cross_split=args.allow_branch_cross_split,
            min_total_for_eval=args.min_total_for_eval,
            require_same_eval_set=args.require_same_eval_set,
        )

        split_prefix = _resolve_split_prefix(out_path, args.split_prefix)
        for split_name in ("train", "valid", "test"):
            split_path = split_prefix.parent / f"{split_prefix.name}-{split_name}.xyz"
            _write_atoms(split_path, splits[split_name])
            split_paths[split_name] = split_path

        split_summary = {
            "split_z": split_z,
            "split_syms": split_syms,
            "z_all": z_all,
            "syms_all": syms_all,
        }

    isolated_path = None
    if args.write_isolated_atoms:
        isolated_e0s = HARDCODED_ISOLATED_E0S if args.isolated_energy_source == "hardcoded" else e0_z
        isolated_atoms = create_isolated_atoms(
            isolated_e0s,
            box=args.isolated_box,
            energy_key="REF_energy",
        )
        isolated_path = Path(args.isolated_out)
        _write_atoms(isolated_path, isolated_atoms)

    ext_path = None
    ext_atoms: Optional[List[Atoms]] = None
    if args.write_ext:
        ext_atoms = fetch_ferroelectric_ext_atoms(
            args.api_key,
            project=args.ext_project,
            max_contributions=args.max_ext_contributions,
            timeout=args.timeout,
            spontaneous_scale=args.ext_spontaneous_scale,
            verbose=args.verbose,
        )
        ext_path = Path(args.ext_out)
        _write_atoms(ext_path, ext_atoms)

    print(f"Done. Wrote {len(atoms_list)} ferroelectric structures to {out_path}.")
    if split_paths:
        for split_name in ("train", "valid", "test"):
            path = split_paths[split_name]
            print(f"  {split_name:>5}: {path}")
        if split_summary is not None:
            for split_name in ("train", "valid", "test"):
                elems = split_summary["split_syms"][split_name]
                print(f"         {split_name} elements: {elems}")
    if isolated_path is not None:
        print(f"  isolated atoms: {isolated_path}")
    if ext_path is not None and ext_atoms is not None:
        print(f"  ferroelectrics_ext: {ext_path} ({len(ext_atoms)} structures)")


if __name__ == "__main__":
    main()
