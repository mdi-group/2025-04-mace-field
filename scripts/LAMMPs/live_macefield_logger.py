#!/usr/bin/env python3
"""Live MACEField response logging for LAMMPS python/invoke callbacks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.io.lammpsrun import construct_cell, lammps_data_to_ase_atoms
from lammps import lammps
from mace.calculators import MACECalculator

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _rank_from_env() -> int:
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MV2_COMM_WORLD_RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def _lammps_rank(lmp: lammps) -> int:
    try:
        comm = lmp.get_mpi_comm()
        if comm is not None:
            return int(comm.Get_rank())
    except Exception:
        pass
    return _rank_from_env()


def is_lammps_root(lammps_ptr) -> bool:
    """Return True only on rank 0 of the active LAMMPS communicator."""
    lmp = lammps(ptr=lammps_ptr)
    return _lammps_rank(lmp) == 0


def _as_array(ctypes_array, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ctypeslib.as_array(ctypes_array)
    return np.array(array, copy=True).reshape(shape)


def _stress_matrix_bar_from_thermo(lmp: lammps) -> np.ndarray:
    return np.asarray(
        [
            [lmp.get_thermo("pxx"), lmp.get_thermo("pxy"), lmp.get_thermo("pxz")],
            [lmp.get_thermo("pxy"), lmp.get_thermo("pyy"), lmp.get_thermo("pyz")],
            [lmp.get_thermo("pxz"), lmp.get_thermo("pyz"), lmp.get_thermo("pzz")],
        ],
        dtype=float,
    )


def _stress_voigt_ase_from_thermo(lmp: lammps) -> np.ndarray:
    stress_bar = np.asarray(
        [
            lmp.get_thermo("pxx"),
            lmp.get_thermo("pyy"),
            lmp.get_thermo("pzz"),
            lmp.get_thermo("pyz"),
            lmp.get_thermo("pxz"),
            lmp.get_thermo("pxy"),
        ],
        dtype=float,
    )
    return np.asarray(convert(stress_bar, "pressure", "metal", "ASE"), dtype=float)


def _build_atoms_from_snapshot(
    *,
    ids: np.ndarray,
    types: np.ndarray,
    positions: np.ndarray,
    forces: np.ndarray,
    boxlo: Sequence[float],
    boxhi: Sequence[float],
    xy: float,
    yz: float,
    xz: float,
    specorder: Sequence[str],
):
    diagdisp = (
        float(boxlo[0]),
        float(boxhi[0]),
        float(boxlo[1]),
        float(boxhi[1]),
        float(boxlo[2]),
        float(boxhi[2]),
    )
    cell, celldisp = construct_cell(diagdisp, (float(xy), float(xz), float(yz)))

    data = np.column_stack((ids, types, positions, forces))

    atoms = lammps_data_to_ase_atoms(
        data=data,
        colnames=["id", "type", "x", "y", "z", "fx", "fy", "fz"],
        cell=cell,
        celldisp=celldisp,
        pbc=True,
        specorder=list(specorder),
        units="metal",
    )
    return atoms


class MultiHeadMACEFieldCalculator:
    """Evaluate different MACEField heads, optionally on different devices."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        polarization_head: str = "pt_head",
        response_head: str = "pt_head",
        energy_head: str = "pt_head",
        polarization_device: Optional[str] = None,
        response_device: Optional[str] = None,
        energy_device: Optional[str] = None,
        enable_cueq: bool = False,
        enable_oeq: bool = False,
        compute_energy: bool = False,
    ) -> None:
        self.compute_energy = compute_energy
        self._electric_field = np.zeros(3, dtype=float)
        self._calculators = {}

        common_kwargs = {
            "model_paths": model_path,
            "default_dtype": dtype,
            "model_type": "MACEField",
            "enable_cueq": enable_cueq,
            "enable_oeq": enable_oeq,
        }

        def get_calculator(head_name: str, device_name: str):
            key = (head_name, device_name)
            calculator = self._calculators.get(key)
            if calculator is None:
                calculator = MACECalculator(
                    head=head_name,
                    device=device_name,
                    **common_kwargs,
                )
                self._calculators[key] = calculator
            return calculator

        polarization_device_name = polarization_device or device
        response_device_name = response_device or device
        energy_device_name = energy_device or device

        self.polarization_calc = get_calculator(
            polarization_head,
            polarization_device_name,
        )
        self.response_calc = get_calculator(
            response_head,
            response_device_name,
        )

        self.energy_calc = None
        if compute_energy:
            self.energy_calc = get_calculator(
                energy_head,
                energy_device_name,
            )

    @property
    def electric_field(self) -> np.ndarray:
        return self._electric_field

    @electric_field.setter
    def electric_field(self, value: Iterable[float]) -> None:
        field = np.asarray(value, dtype=float).reshape(3)
        self._electric_field = field
        if self.energy_calc is not None:
            self.energy_calc.electric_field = field
        self.polarization_calc.electric_field = field
        self.response_calc.electric_field = field

    def evaluate(self, atoms) -> dict:
        results = {
            "polarization": np.asarray(
                self.polarization_calc.get_property("polarization", atoms)
            ).reshape(3),
            "becs": np.asarray(
                self.response_calc.get_property("becs", atoms)
            ).reshape(len(atoms), 9),
            "polarizability": np.asarray(
                self.response_calc.get_property("polarizability", atoms)
            ).reshape(9),
        }

        if self.energy_calc is not None:
            results.update(
                {
                    "energy": self.energy_calc.get_property("energy", atoms),
                    "forces": self.energy_calc.get_property("forces", atoms),
                    "stress": self.energy_calc.get_property("stress", atoms),
                }
            )

        return results

class LiveMACEFieldLogger:
    """Write live response-property snapshots from a running LAMMPS simulation."""

    def __init__(
        self,
        *,
        model_path: str,
        specorder: Sequence[str],
        output_path: str,
        scalar_output_path: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        polarization_head: str = "pt_head",
        response_head: str = "pt_head",
        energy_head: str = "pt_head",
        polarization_device: Optional[str] = None,
        response_device: Optional[str] = None,
        energy_device: Optional[str] = None,
        enable_cueq: bool = False,
        enable_oeq: bool = False,
        compute_energy: bool = False,
    ) -> None:
        self.specorder = list(specorder)
        self.output_path = Path(output_path)
        self.scalar_output_path = Path(scalar_output_path) if scalar_output_path else None
        self.first_frame = True
        self.compute_energy = compute_energy
        self.calc = MultiHeadMACEFieldCalculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            polarization_head=polarization_head,
            response_head=response_head,
            energy_head=energy_head,
            polarization_device=polarization_device or None,
            response_device=response_device or None,
            energy_device=energy_device or None,
            enable_cueq=enable_cueq,
            enable_oeq=enable_oeq,
            compute_energy=compute_energy,
        )

        if self.output_path.exists():
            self.output_path.unlink()
        if self.scalar_output_path is not None and self.scalar_output_path.exists():
            self.scalar_output_path.unlink()

    def _atoms_from_lammps(self, lmp: lammps):
        natoms = int(lmp.get_natoms())
        ids = _as_array(lmp.gather_atoms("id", 0, 1), (natoms,))
        types = _as_array(lmp.gather_atoms("type", 0, 1), (natoms,))
        positions = _as_array(lmp.gather_atoms("x", 1, 3), (natoms, 3))
        forces = _as_array(lmp.gather_atoms("f", 1, 3), (natoms, 3))
        sort_order = np.argsort(ids)
        ids = ids[sort_order]
        types = types[sort_order]
        positions = positions[sort_order]
        forces = forces[sort_order]

        boxlo, boxhi, xy, yz, xz, _periodicity, _box_change = lmp.extract_box()
        atoms = _build_atoms_from_snapshot(
            ids=ids,
            types=types,
            positions=positions,
            forces=forces,
            boxlo=boxlo,
            boxhi=boxhi,
            xy=xy,
            yz=yz,
            xz=xz,
            specorder=self.specorder,
        )

        energy = float(convert(float(lmp.get_thermo("pe")), "energy", "metal", "ASE"))
        forces_ase = np.asarray(convert(forces, "force", "metal", "ASE"), dtype=float)
        stress = _stress_voigt_ase_from_thermo(lmp)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=forces_ase,
            stress=stress,
        )
        return atoms

    def _field_from_lammps(self, lmp: lammps) -> np.ndarray:
        return np.asarray(
            [
                float(lmp.extract_variable("Ex", None, 0)),
                float(lmp.extract_variable("Ey", None, 0)),
                float(lmp.extract_variable("Ez", None, 0)),
            ],
            dtype=float,
        )

    def _append_scalar_row(self, *, lmp: lammps, field: np.ndarray, results: dict) -> None:
        if self.scalar_output_path is None:
            return

        scalar_path = self.scalar_output_path
        if not scalar_path.exists():
            header = (
                "# step time_ps temp_K pe_lammps press_bar "
                "Ex Ey Ez "
                "Px Py Pz "
                "alpha_xx alpha_xy alpha_xz alpha_yx alpha_yy alpha_yz alpha_zx alpha_zy alpha_zz"
            )
            with scalar_path.open("w", encoding="utf-8") as handle:
                handle.write(header + "\n")

        row = [
            int(round(lmp.get_thermo("step"))),
            float(lmp.get_thermo("time")),
            float(lmp.get_thermo("temp")),
            float(lmp.get_thermo("pe")),
            float(lmp.get_thermo("press")),
            *field.tolist(),
            *np.asarray(results["polarization"]).reshape(3).tolist(),
            *np.asarray(results["polarizability"]).reshape(9).tolist(),
        ]
        with scalar_path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(str(value) for value in row) + "\n")

    def log_step(self, lammps_ptr) -> None:
        lmp = lammps(ptr=lammps_ptr)
        atoms = self._atoms_from_lammps(lmp)
        field = self._field_from_lammps(lmp)

        self.calc.electric_field = field
        results = self.calc.evaluate(atoms)

        atoms.info["step"] = int(round(lmp.get_thermo("step")))
        atoms.info["time_ps"] = float(lmp.get_thermo("time"))
        atoms.info["temperature_K"] = float(lmp.get_thermo("temp"))
        atoms.info["lammps_pressure_bar"] = float(lmp.get_thermo("press"))
        atoms.info["lammps_stress_bar"] = _stress_matrix_bar_from_thermo(lmp)
        atoms.info["MACE_electric_field"] = field
        atoms.info["MACE_polarization"] = results["polarization"]
        atoms.info["MACE_polarizability"] = results["polarizability"]
        atoms.arrays["MACE_becs"] = results["becs"]

        if "energy" in results and results["energy"] is not None:
            atoms.info["MACE_energy"] = float(results["energy"])
        if "stress" in results and results["stress"] is not None:
            atoms.info["MACE_stress"] = np.asarray(results["stress"]).reshape(-1)
        if "forces" in results and results["forces"] is not None:
            atoms.arrays["MACE_forces"] = np.asarray(results["forces"])

        write(
            str(self.output_path),
            atoms,
            format="extxyz",
            append=not self.first_frame,
        )
        self._append_scalar_row(lmp=lmp, field=field, results=results)
        self.first_frame = False
