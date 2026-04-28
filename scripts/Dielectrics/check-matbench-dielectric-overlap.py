#!/usr/bin/env python3
"""
Cross-check structural overlap between a local MP dielectric dataset and the
official Matbench dielectric benchmark dataset.

This script compares a local `.extxyz`/`.xyz` file such as `MP-Dielectrics.extxyz`
against the official Matbench dielectric benchmark payload from
`https://ml.materialsproject.org/projects/matbench_dielectric.json.gz`.

Because the Matbench dataset only exposes `structure` and the target `n`
(refractive index), overlap is determined by exact-ish structural matching
within reduced-formula buckets using pymatgen's StructureMatcher.

Requirements:
  pip install ase pymatgen requests tqdm

Example:
  python check-matbench-dielectric-overlap.py \
    --mp-dataset MP-Dielectrics.extxyz

  python check-matbench-dielectric-overlap.py \
    --mp-dataset MP-Dielectrics-filtered.extxyz \
    --csv-out filtered-vs-matbench-overlap.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MATBENCH_DIELECTRIC_URL = "https://ml.materialsproject.org/projects/matbench_dielectric.json.gz"
DEFAULT_CACHE_NAME = "matbench_dielectric.json.gz"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Codex overlap checker)"
SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_output_path(base_path: Path, requested: str, suffix: str, ext: str) -> Path:
    if requested.strip():
        path = Path(requested)
        return path if path.is_absolute() else base_path.parent / path
    return base_path.with_name(f"{base_path.stem}{suffix}{ext}")


def _read_atoms_list(path: Path) -> List[Any]:
    atoms = ase_read(path, index=":")
    return atoms if isinstance(atoms, list) else [atoms]


def _load_local_mp_records(path: Path) -> List[Dict[str, Any]]:
    adaptor = AseAtomsAdaptor()
    atoms_list = _read_atoms_list(path)
    records: List[Dict[str, Any]] = []

    for idx, atoms in enumerate(atoms_list):
        structure = adaptor.get_structure(atoms)
        records.append(
            {
                "source_index": idx,
                "material_id": atoms.info.get("material_id"),
                "dielectric_task_id": atoms.info.get("dielectric_task_id"),
                "formula": structure.composition.reduced_formula,
                "nsites": len(structure),
                "structure": structure,
            }
        )
    return records


def _download_matbench_payload(cache_path: Path, refresh: bool, timeout: float) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        MATBENCH_DIELECTRIC_URL,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.content


def _load_matbench_records(cache_path: Path, refresh: bool, timeout: float) -> List[Dict[str, Any]]:
    payload = _download_matbench_payload(cache_path=cache_path, refresh=refresh, timeout=timeout)
    decoded = gzip.decompress(payload).decode("utf-8")
    raw = json.loads(decoded)

    if not isinstance(raw, dict) or set(raw.keys()) != {"index", "columns", "data"}:
        raise ValueError("Unexpected Matbench dielectric payload format.")

    if raw["columns"] != ["structure", "n"]:
        raise ValueError(f"Unexpected Matbench dielectric columns: {raw['columns']}")

    records: List[Dict[str, Any]] = []
    for benchmark_index, row in zip(raw["index"], raw["data"]):
        structure = Structure.from_dict(row[0])
        records.append(
            {
                "benchmark_index": int(benchmark_index),
                "matbench_n": float(row[1]),
                "formula": structure.composition.reduced_formula,
                "nsites": len(structure),
                "structure": structure,
            }
        )
    return records


def _bucket_by_formula(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["formula"])].append(record)
    return buckets


def _overlap_pairs_and_summary(
    mp_records: Sequence[Dict[str, Any]],
    matbench_records: Sequence[Dict[str, Any]],
    *,
    ltol: float,
    stol: float,
    angle_tol: float,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    mp_by_formula = _bucket_by_formula(mp_records)
    matbench_by_formula = _bucket_by_formula(matbench_records)
    common_formulas = sorted(set(mp_by_formula) & set(matbench_by_formula))

    overlap_pairs: List[Dict[str, Any]] = []
    overlap_group_counter: Counter[str] = Counter()
    mp_overlap_indices: set[int] = set()
    mp_overlap_material_ids: set[str] = set()
    matbench_overlap_indices: set[int] = set()
    n_overlap_groups = 0

    iterator: Iterable[str] = common_formulas
    if tqdm is not None:
        iterator = tqdm(common_formulas, desc="Match structures", unit="formula")

    for formula in iterator:
        combined_structures = []
        record_map: Dict[int, Tuple[str, Dict[str, Any]]] = {}

        for record in mp_by_formula[formula]:
            structure = record["structure"]
            combined_structures.append(structure)
            record_map[id(structure)] = ("mp", record)

        for record in matbench_by_formula[formula]:
            structure = record["structure"]
            combined_structures.append(structure)
            record_map[id(structure)] = ("matbench", record)

        grouped = matcher.group_structures(combined_structures)
        for group in grouped:
            mp_group = [record_map[id(structure)][1] for structure in group if record_map[id(structure)][0] == "mp"]
            matbench_group = [
                record_map[id(structure)][1] for structure in group if record_map[id(structure)][0] == "matbench"
            ]
            if not mp_group or not matbench_group:
                continue

            n_overlap_groups += 1
            overlap_group_counter[formula] += 1

            for mp_record in mp_group:
                mp_overlap_indices.add(int(mp_record["source_index"]))
                material_id = mp_record.get("material_id")
                if material_id is not None:
                    mp_overlap_material_ids.add(str(material_id))

            for matbench_record in matbench_group:
                matbench_overlap_indices.add(int(matbench_record["benchmark_index"]))

            for mp_record in mp_group:
                for matbench_record in matbench_group:
                    overlap_pairs.append(
                        {
                            "formula": formula,
                            "mp_source_index": int(mp_record["source_index"]),
                            "mp_material_id": mp_record.get("material_id"),
                            "mp_dielectric_task_id": mp_record.get("dielectric_task_id"),
                            "mp_nsites": int(mp_record["nsites"]),
                            "matbench_index": int(matbench_record["benchmark_index"]),
                            "matbench_n": float(matbench_record["matbench_n"]),
                            "matbench_nsites": int(matbench_record["nsites"]),
                        }
                    )

    summary = {
        "matbench_url": MATBENCH_DIELECTRIC_URL,
        "n_mp_records": len(mp_records),
        "n_mp_unique_material_ids": len({str(r['material_id']) for r in mp_records if r.get("material_id") is not None}),
        "n_matbench_records": len(matbench_records),
        "n_common_formula_buckets": len(common_formulas),
        "n_overlap_groups": n_overlap_groups,
        "n_overlap_pairs": len(overlap_pairs),
        "n_mp_overlap_records": len(mp_overlap_indices),
        "n_mp_overlap_material_ids": len(mp_overlap_material_ids),
        "n_matbench_overlap_records": len(matbench_overlap_indices),
        "fraction_mp_overlap_records": (len(mp_overlap_indices) / len(mp_records)) if mp_records else 0.0,
        "fraction_matbench_overlap_records": (len(matbench_overlap_indices) / len(matbench_records)) if matbench_records else 0.0,
        "matcher": {
            "ltol": ltol,
            "stol": stol,
            "angle_tol": angle_tol,
        },
        "top_overlap_formulas": [
            {"formula": formula, "n_overlap_groups": count}
            for formula, count in overlap_group_counter.most_common(20)
        ],
    }

    if verbose:
        summary["overlap_formula_counts_full"] = dict(sorted(overlap_group_counter.items()))

    return overlap_pairs, summary


def _write_pairs_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "formula",
        "mp_source_index",
        "mp_material_id",
        "mp_dielectric_task_id",
        "mp_nsites",
        "matbench_index",
        "matbench_n",
        "matbench_nsites",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mp-dataset",
        default="MP-Dielectrics.extxyz",
        help="Local MP dielectric dataset to compare against Matbench dielectric.",
    )
    parser.add_argument(
        "--matbench-cache",
        default=DEFAULT_CACHE_NAME,
        help="Path to cache the official Matbench dielectric json.gz payload.",
    )
    parser.add_argument(
        "--refresh-matbench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-download the Matbench dielectric payload even if a cache file exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for downloading the Matbench dielectric payload.",
    )
    parser.add_argument("--matcher-ltol", type=float, default=0.2, help="StructureMatcher lattice tolerance.")
    parser.add_argument("--matcher-stol", type=float, default=0.3, help="StructureMatcher site tolerance.")
    parser.add_argument("--matcher-angle-tol", type=float, default=5.0, help="StructureMatcher angle tolerance in degrees.")
    parser.add_argument(
        "--csv-out",
        default="",
        help="CSV path for matched MP/Matbench pairs. Defaults to '<mp-dataset>-vs-matbench-dielectric-overlap.csv'.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="JSON path for the summary report. Defaults to '<mp-dataset>-vs-matbench-dielectric-overlap.json'.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mp_dataset_path = Path(args.mp_dataset)
    if not mp_dataset_path.is_absolute():
        mp_dataset_path = SCRIPT_DIR / mp_dataset_path
    if not mp_dataset_path.exists():
        raise SystemExit(f"MP dielectric dataset not found: {mp_dataset_path}")

    cache_path = Path(args.matbench_cache)
    if not cache_path.is_absolute():
        cache_path = mp_dataset_path.parent / cache_path

    csv_out = _resolve_output_path(mp_dataset_path, args.csv_out, "-vs-matbench-dielectric-overlap", ".csv")
    json_out = _resolve_output_path(mp_dataset_path, args.json_out, "-vs-matbench-dielectric-overlap", ".json")

    mp_records = _load_local_mp_records(mp_dataset_path)
    matbench_records = _load_matbench_records(
        cache_path=cache_path,
        refresh=args.refresh_matbench,
        timeout=args.timeout,
    )
    overlap_pairs, summary = _overlap_pairs_and_summary(
        mp_records=mp_records,
        matbench_records=matbench_records,
        ltol=args.matcher_ltol,
        stol=args.matcher_stol,
        angle_tol=args.matcher_angle_tol,
        verbose=args.verbose,
    )

    summary["mp_dataset"] = str(mp_dataset_path)
    summary["matbench_cache"] = str(cache_path)
    summary["csv_out"] = str(csv_out)
    summary["json_out"] = str(json_out)

    _write_pairs_csv(csv_out, overlap_pairs)
    _write_summary_json(json_out, summary)

    print(f"MP dataset:          {mp_dataset_path}")
    print(f"Matbench cache:      {cache_path}")
    print(f"Matched pair rows:   {summary['n_overlap_pairs']}")
    print(f"Overlap groups:      {summary['n_overlap_groups']}")
    print(
        "MP overlaps:         "
        f"{summary['n_mp_overlap_records']} / {summary['n_mp_records']} frames "
        f"({summary['fraction_mp_overlap_records']:.1%})"
    )
    print(
        "Matbench overlaps:   "
        f"{summary['n_matbench_overlap_records']} / {summary['n_matbench_records']} rows "
        f"({summary['fraction_matbench_overlap_records']:.1%})"
    )
    print(f"Pair CSV:            {csv_out}")
    print(f"Summary JSON:        {json_out}")


if __name__ == "__main__":
    main()
