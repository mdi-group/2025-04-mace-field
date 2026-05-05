#!/usr/bin/env python3
"""Convert a LAMMPS dump plus a sidecar thermo table into extxyz.

The intended workflow is:
1. dump per-atom coordinates and forces with ``dump custom``
2. write matching per-frame thermo data to a whitespace-delimited file
3. use this script to merge them into one extxyz trajectory

The output extxyz keeps:
- per-atom forces from the LAMMPS dump
- potential energy from the thermo table
- electric field and stress metadata from the thermo table
"""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write


def _parse_thermo_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    header: List[str] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if header is None:
                if line.startswith("#"):
                    line = line[1:].strip()
                header = line.split()
                if not header:
                    raise ValueError(f"Missing thermo header in {path} at line {lineno}")
                continue

            parts = line.split()
            if len(parts) != len(header):
                raise ValueError(
                    f"Thermo column count mismatch in {path} at line {lineno}: "
                    f"expected {len(header)}, got {len(parts)}"
                )

            row: Dict[str, float] = {}
            for key, value in zip(header, parts):
                row[key] = float(value)
            rows.append(row)

    if header is None:
        raise ValueError(f"No thermo data found in {path}")

    return rows


def _iter_frames(dump_path: Path, specorder: Iterable[str]) -> Iterator:
    return iread(
        str(dump_path),
        format="lammps-dump-text",
        index=":",
        specorder=list(specorder),
    )


def _stress_tensor_bar(row: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            row["pxx"],
            row["pxy"],
            row["pxz"],
            row["pxy"],
            row["pyy"],
            row["pyz"],
            row["pxz"],
            row["pyz"],
            row["pzz"],
        ],
        dtype=float,
    )


def _attach_metadata(atoms, row: Dict[str, float]) -> None:
    forces = atoms.get_forces()
    potential_energy = float(row["pe"])

    atoms.calc = SinglePointCalculator(
        atoms,
        energy=potential_energy,
        forces=forces,
    )

    atoms.info["timestep"] = int(round(row["step"]))
    atoms.info["time_ps"] = float(row["time"])
    atoms.info["lammps_kinetic_energy"] = float(row["ke"])
    atoms.info["lammps_total_energy"] = float(row["etotal"])
    atoms.info["temperature_K"] = float(row["temp"])
    atoms.info["lammps_pressure_bar"] = float(row["press"])
    atoms.info["lammps_stress_bar"] = _stress_tensor_bar(row)
    atoms.info["MACE_electric_field"] = np.asarray(
        [row.get("Ex", 0.0), row.get("Ey", 0.0), row.get("Ez", 0.0)],
        dtype=float,
    )


def convert(
    dump_path: Path,
    thermo_path: Path,
    output_path: Path,
    specorder: Iterable[str],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    thermo_rows = _parse_thermo_rows(thermo_path)
    frames = _iter_frames(dump_path, specorder)

    wrote_any = False
    frame_count = 0
    for frame_count, pair in enumerate(zip_longest(frames, thermo_rows), start=1):
        atoms, row = pair
        if atoms is None or row is None:
            raise ValueError(
                f"Frame/thermo mismatch near frame {frame_count}: "
                f"dump={dump_path}, thermo={thermo_path}"
            )
        _attach_metadata(atoms, row)
        write(
            str(output_path),
            atoms,
            format="extxyz",
            append=wrote_any,
        )
        wrote_any = True

    if frame_count == 0:
        raise ValueError(f"No frames were read from {dump_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a LAMMPS custom dump and a sidecar thermo table into extxyz."
    )
    parser.add_argument(
        "--dump",
        required=True,
        help="LAMMPS dump file written by dump custom.",
    )
    parser.add_argument(
        "--thermo",
        required=True,
        help="Whitespace-delimited thermo file with one header row and one row per dumped frame.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output extxyz trajectory.",
    )
    parser.add_argument(
        "--specorder",
        nargs="+",
        required=True,
        help="LAMMPS type-to-element mapping, e.g. --specorder O Si.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dump_path = Path(args.dump).expanduser().resolve()
    thermo_path = Path(args.thermo).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not dump_path.exists():
        raise FileNotFoundError(f"LAMMPS dump not found: {dump_path}")
    if not thermo_path.exists():
        raise FileNotFoundError(f"Thermo table not found: {thermo_path}")

    convert(
        dump_path=dump_path,
        thermo_path=thermo_path,
        output_path=output_path,
        specorder=args.specorder,
        overwrite=args.overwrite,
    )
    print(f"Wrote extxyz trajectory: {output_path}")


if __name__ == "__main__":
    main()
