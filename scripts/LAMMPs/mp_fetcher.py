#!/usr/bin/env python3
"""
mp_fetcher.py

Fetch a structure from Materials Project, build a supercell, and write:
  - optional extended XYZ (ASE extxyz, preserves cell/pbc)
  - LAMMPS data file (read_data) with deterministic type<->element mapping
  - optional LAMMPS input file configured for ML-IAP (mliap unified) + MACEField E-field ramp

Examples:
  export MP_API_KEY="J0JE22k0f7XptxXUH0ofSLr0XQTRuIIm"
  python mp_fetcher.py fetch mp-149 --supercell 2 2 2 --out-dir MD
  python mp_fetcher.py fetch mp-149 --no-xyz --json

  # customize model + MD params
  python mp_fetcher.py fetch mp-7000 \
    --mace-model models/MACEField-omat-dielectric.model-mliap_lammps.pt \
    --E0 0.05 --period 10000 --nseg 100 --run 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from mp_api.client import MPRester

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pymatgen is required for MaterialsFetcher -> ASE conversion. "
        "Install with: pip install pymatgen"
    ) from exc


def _sorted_unique_elements(symbols: Iterable[str]) -> List[str]:
    """Stable, model-friendly ordering by atomic number."""
    return sorted(set(symbols), key=lambda s: atomic_numbers[s])


def _parse_3ints(vals: Sequence[str], *, what: str) -> Tuple[int, int, int]:
    if len(vals) != 3:
        raise ValueError(f"{what} must have exactly 3 integers, e.g. --{what} 2 2 2")
    try:
        a, b, c = (int(v) for v in vals)
    except ValueError as e:
        raise ValueError(f"{what} values must be integers") from e
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"{what} values must be positive integers")
    return a, b, c


def _mass_for_symbol(sym: str) -> float:
    z = atomic_numbers[sym]
    m = float(atomic_masses[z])
    if m <= 0:
        raise ValueError(f"Could not determine atomic mass for element {sym}")
    return m


def render_macefield_infile(
    *,
    title: str,
    data_filename: str,
    elements: List[str],
    counts_by_element: dict[str, int],
    atom_style: str = "atomic",
    units: str = "metal",
    boundary: str = "p p p",
    newton: str = "on",
    tilt_large: bool = True,
    replicate: Tuple[int, int, int] = (1, 1, 1),
    mace_model: str = "mace-field-mp-0b3-medium-mh-asr-EF-200.model-mliap_lammps.pt",
    mliap_flags: str = "0",
    neighbor: float = 2.0,
    timestep: float = 0.001,
    temperature: float = 300.0,
    tdamp: float = 0.1,
    thermo: int = 10,
    dump_every: int = 10,
    dump_filename: str = "traj.lammpstrj",
    run_steps: int = 100000,
    # E-field ramp params
    E0: float = 0.05,
    period: int = 10000,
    nseg: int = 100,
    Ex: float = 0.0,
    Ey: float = 0.0,
) -> str:
    # Build a deterministic "type i = Element (count)" comment block
    type_lines = []
    for i, el in enumerate(elements, start=1):
        n = counts_by_element.get(el, 0)
        type_lines.append(f"# type {i} = {el} ({n})")

    # Mass lines follow specorder mapping
    mass_lines = []
    for i, el in enumerate(elements, start=1):
        mass_lines.append(f"mass {i} {_mass_for_symbol(el):.10g}")

    rep_a, rep_b, rep_c = replicate

    lines: List[str] = []
    lines.append(f"# {title} + MACEField via ML-IAP (unified)")
    lines.append("# Example run (edit to taste):")
    lines.append(
        f"#   lmp_mpi -np 4 -in in.{title}_macefield"
    )
    lines.append("")
    lines.append(f"units           {units}")
    lines.append(f"atom_style      {atom_style}")
    lines.append(f"boundary        {boundary}")
    lines.append(f"newton          {newton}")
    lines.append("")
    if tilt_large:
        lines.append("# (optional but harmless; avoids 'tilt too large' limits if you ever strain the cell)")
        lines.append("box             tilt large")
        lines.append("")
    lines.append(f"read_data       {data_filename}")
    lines.append(f"replicate       {rep_a} {rep_b} {rep_c}")
    lines.append("")
    lines.append("# Atom types from your data file counts:")
    lines.extend(type_lines)
    lines.append("")
    lines.extend(mass_lines)
    lines.append("")
    lines.append("# ---- Pair style (ML-IAP unified) ----")
    lines.append(f"pair_style      mliap unified {mace_model} {mliap_flags}")
    lines.append(f"pair_coeff      * * {' '.join(elements)}")
    lines.append("")
    lines.append(f"neighbor        {neighbor} bin")
    lines.append("neigh_modify    every 1 delay 0 check yes")
    lines.append("")
    lines.append("# Debug: prints whatever is in the environment at startup")
    lines.append("variable M getenv MACE_EFIELD")
    lines.append('print "LAMMPS getenv MACE_EFIELD (startup) = ${M}"')
    lines.append("")
    lines.append("# ---- Time-dependent electric field (ramp 0 -> E0 over 'period' steps, then stays at E0) ----")
    lines.append(f"variable        E0     equal {E0}")
    lines.append(f"variable        period equal {period}")
    lines.append(f"variable        nseg   equal {nseg}")
    lines.append("")
    lines.append(f"variable        Ex     equal {Ex}")
    lines.append(f"variable        Ey     equal {Ey}")
    lines.append("")
    lines.append("# k increases by 1 every (period/nseg) steps; after one period, Ez stays at E0 due to clamping")
    lines.append("variable        k      equal floor(v_nseg*(step+1)/v_period)")
    lines.append("variable        raw    equal v_k/v_nseg")
    lines.append("variable        frac   equal v_raw - (v_raw>1)*(v_raw-1)")
    lines.append("variable        Ez     equal v_E0*v_frac")
    lines.append("")
    lines.append('python set_mace_efield here """')
    lines.append("import os")
    lines.append("from lammps import lammps")
    lines.append("")
    lines.append("def set_mace_efield(lammps_ptr):")
    lines.append("    lmp = lammps(ptr=lammps_ptr)")
    lines.append('    ex = float(lmp.extract_variable("Ex", None, 0))')
    lines.append('    ey = float(lmp.extract_variable("Ey", None, 0))')
    lines.append('    ez = float(lmp.extract_variable("Ez", None, 0))')
    lines.append('    os.environ["MACE_EFIELD"] = f"{ex},{ey},{ez}"')
    lines.append('"""')
    lines.append("fix mace_efield all python/invoke 1 end_of_step set_mace_efield")
    lines.append("")
    lines.append("# ---- MD ----")
    lines.append(f"timestep        {timestep}")
    lines.append("reset_timestep  0")
    lines.append("")
    lines.append(f"variable        T equal {temperature}")
    lines.append("velocity        all create ${T} 4928459 mom yes dist gaussian")
    lines.append(f"fix             nvt_all all nvt temp ${{T}} ${{T}} {tdamp}")
    lines.append("")
    lines.append("# ---- Output ----")
    lines.append(f"thermo          {thermo}")
    lines.append("thermo_style    custom step time temp pe ke etotal press vol v_Ex v_Ey v_Ez")
    lines.append("")
    lines.append(
        f"dump            traj all custom {dump_every} {dump_filename} id type xu yu zu fx fy fz"
    )
    lines.append("dump_modify     traj sort id")
    lines.append("")
    lines.append(f"run             {run_steps}")
    lines.append("")

    return "\n".join(lines)


@dataclass
class FetchResult:
    name: str
    dir: str
    xyz: Optional[str]
    lammps_data: str
    lammps_in: Optional[str]
    elements: List[str]
    atoms: Optional[Atoms] = None  # not JSON-serializable; keep optional for API users


class MaterialFetcher:
    """
    API wrapper around mp_api + pymatgen + ASE.

    Notes:
      - LAMMPS 'type' order is deterministic via specorder=elements.
      - Directory is always created; LAMMPS data is always written.
      - XYZ and input file writing are optional.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "Missing Materials Project API key. Provide api_key or set MP_API_KEY."
            )
        self.api_key = api_key

    def fetch(
        self,
        mpid: str,
        supercell: Tuple[int, int, int] = (2, 2, 2),
        out_dir: str = "MD",
        filename_stem: Optional[str] = None,
        atom_style: str = "atomic",
        *,
        write_xyz: bool = True,
        write_infile: bool = True,
        return_atoms: bool = False,
        # infile params
        mace_model: str = "models/MACEField-omat-dielectric.model-mliap_lammps.pt",
        period: int = 10000,
        nseg: int = 100,
        Ex: float = 0.0,
        Ey: float = 0.0,
        E0: float = 0.0,
        replicate: Tuple[int, int, int] = (1, 1, 1),
        timestep: float = 0.001,
        temperature: float = 300.0,
        run_steps: int = 100000,
        dump_every: int = 10,
        thermo: int = 10,
    ) -> FetchResult:
        # 1) Download and supercell (pymatgen Structure)
        with MPRester(api_key=self.api_key) as mpr:
            struct = mpr.get_structure_by_material_id(mpid)

        struct.make_supercell(supercell)

        formula = struct.composition.reduced_formula
        name = filename_stem or f"{formula}-{mpid}"
        target_dir = os.path.join(out_dir, name)
        os.makedirs(target_dir, exist_ok=True)

        # 2) Convert to ASE atoms
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.pbc = True

        # 3) Determine element/type order used for LAMMPS
        elements = _sorted_unique_elements(atoms.get_chemical_symbols())

        # Counts for nice comments in the infile
        c = Counter(atoms.get_chemical_symbols())
        counts_by_element = {el: int(c.get(el, 0)) for el in elements}

        # 4) Write extended XYZ (optional)
        xyz_path: Optional[str] = None
        if write_xyz:
            xyz_path = os.path.join(target_dir, f"{name}.xyz")
            write(xyz_path, atoms, format="extxyz")

        # 5) Write LAMMPS data file with deterministic type mapping (always)
        data_filename = "structure.data"
        data_path = os.path.join(target_dir, data_filename)
        write(
            data_path,
            atoms,
            format="lammps-data",
            atom_style=atom_style,
            specorder=elements,
        )

        # 6) Write LAMMPS infile (optional)
        in_path: Optional[str] = None
        if write_infile:
            in_filename = f"in.{name}_macefield"
            in_path = os.path.join(target_dir, in_filename)
            dump_filename = f"{name}.lammpstrj"
            text = render_macefield_infile(
                title=name,
                data_filename=data_filename,
                elements=elements,
                counts_by_element=counts_by_element,
                atom_style=atom_style,
                mace_model=mace_model,
                replicate=replicate,
                timestep=timestep,
                temperature=temperature,
                run_steps=run_steps,
                dump_every=dump_every,
                thermo=thermo,
                dump_filename=dump_filename,
                E0=E0,
                period=period,
                nseg=nseg,
                Ex=Ex,
                Ey=Ey,
            )
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(text)

        return FetchResult(
            name=name,
            dir=target_dir,
            xyz=xyz_path,
            lammps_data=data_path,
            lammps_in=in_path,
            elements=elements,
            atoms=atoms if return_atoms else None,
        )


def _result_to_jsonable(r: FetchResult) -> dict:
    d = asdict(r)
    d["atoms"] = None  # Atoms cannot be JSON-serialized
    return d


def _cmd_fetch(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("MP_API_KEY", "")
    fetcher = MaterialFetcher(api_key=api_key)

    res = fetcher.fetch(
        mpid=args.mpid,
        supercell=_parse_3ints(args.supercell, what="supercell"),
        out_dir=args.out_dir,
        filename_stem=args.name,
        atom_style=args.atom_style,
        write_xyz=not args.no_xyz,
        write_infile=not args.no_infile,
        # infile params
        mace_model=args.mace_model,
        period=args.period,
        nseg=args.nseg,
        Ex=args.Ex,
        Ey=args.Ey,
        E0=args.Ez,
        replicate=_parse_3ints(args.replicate, what="replicate"),
        timestep=args.timestep,
        temperature=args.temperature,
        run_steps=args.run,
        dump_every=args.dump_every,
        thermo=args.thermo,
    )

    if args.json:
        print(json.dumps(_result_to_jsonable(res), indent=2))
    else:
        print(f"Name:        {res.name}")
        print(f"Directory:   {res.dir}")
        print(f"Elements:    {', '.join(res.elements)}  (LAMMPS types 1..N in this order)")
        print(f"LAMMPS data:  {res.lammps_data}")
        print(f"XYZ:         {res.xyz if res.xyz else '(not written)'}")
        print(f"Infile:      {res.lammps_in if res.lammps_in else '(not written)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mp_fetcher",
        description="Fetch a Materials Project structure, build a supercell, write XYZ + LAMMPS data (+ optional MACEField infile).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="Fetch a material by Materials Project ID (e.g. mp-149).")
    pf.add_argument("mpid", help="Materials Project material ID (e.g. mp-149)")
    pf.add_argument("--api-key", default=None, help="MP API key (or set MP_API_KEY env var).")
    pf.add_argument(
        "--supercell",
        nargs=3,
        default=("2", "2", "2"),
        metavar=("A", "B", "C"),
        help="Supercell size, e.g. --supercell 2 2 2",
    )
    pf.add_argument("--out-dir", default="MD", help="Base output directory (default: MD)")
    pf.add_argument("--name", default=None, help="Filename stem / directory name (default: <formula>-<mpid>)")
    pf.add_argument("--atom-style", default="atomic", help="LAMMPS atom_style (default: atomic)")

    pf.add_argument("--no-xyz", action="store_true", help="Do not write extended XYZ")

    # Infile controls
    pf.add_argument("--no-infile", action="store_true", help="Do not write LAMMPS input file")
    pf.add_argument(
        "--mace-model",
        default="models/MACEField-omat-dielectric.model-mliap_lammps.pt",
        help="Path/name of the MACEField ML-IAP model file",
    )
    pf.add_argument("--replicate", nargs=3, default=("1", "1", "1"), metavar=("A", "B", "C"),
                    help="replicate A B C in the LAMMPS input (default: 1 1 1)")
    pf.add_argument("--timestep", type=float, default=0.001, help="MD timestep (default: 0.001)")
    pf.add_argument("--temperature", type=float, default=300.0, help="NVT temperature (default: 300)")
    pf.add_argument("--run", type=int, default=100000, help="Run steps (default: 100000)")
    pf.add_argument("--thermo", type=int, default=10, help="Thermo frequency (default: 10)")
    pf.add_argument("--dump-every", type=int, default=10, help="Dump frequency (default: 10)")

    # E-field ramp params
    pf.add_argument("--period", type=int, default=10000, help="Ramp period in steps (default: 10000)")
    pf.add_argument("--nseg", type=int, default=100, help="Ramp segments (default: 100)")
    pf.add_argument("--Ex", type=float, default=0.0, help="Constant Ex (default: 0.0)")
    pf.add_argument("--Ey", type=float, default=0.0, help="Constant Ey (default: 0.0)")
    pf.add_argument("--Ez", type=float, default=0.0, help="Constant Ez (default: 0.0)")

    pf.add_argument("--json", action="store_true", help="Print machine-readable JSON result")

    pf.set_defaults(func=_cmd_fetch)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
