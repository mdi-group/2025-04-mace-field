#!/usr/bin/env python3
"""Backfill MACEField response properties into an extxyz trajectory.

This is designed for LAMMPS/ML-IAP trajectories that already contain the MD
positions, velocities, forces, energies, and stress, but did not write the
MACEField response quantities. By default the script only computes the missing
response properties to keep GPU memory usage down:

- ``MACE_electric_field`` (info)
- ``MACE_polarization`` (info)
- ``MACE_polarizability`` (info, flattened 3x3 tensor)
- ``MACE_becs`` (per-atom array with shape ``(natoms, 9)``)

It can optionally recompute ``MACE_energy``, ``MACE_stress``, and
``MACE_forces`` too, but that is disabled by default because it is the most
memory-intensive path and is usually unnecessary for postprocessing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from ase.io import iread, write
from tqdm import tqdm

from mace.calculators import MACECalculator

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class UnifiedMACEFieldCalculator:
    """Evaluate all requested MACEField properties from one head/device."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        head: str = "pt_head",
        enable_cueq: bool = False,
        enable_oeq: bool = False,
        compute_energy: bool = False,
    ) -> None:
        self.compute_energy = compute_energy
        self._electric_field = np.zeros(3, dtype=float)

        common_kwargs = {
            "model_paths": model_path,
            "default_dtype": dtype,
            "model_type": "MACEField",
            "enable_cueq": enable_cueq,
            "enable_oeq": enable_oeq,
        }

        self.calc = MACECalculator(
            head=head,
            device=device,
            **common_kwargs,
        )

    @property
    def electric_field(self) -> np.ndarray:
        return self._electric_field

    @electric_field.setter
    def electric_field(self, value: Iterable[float]) -> None:
        field = np.asarray(value, dtype=float).reshape(3)
        self._electric_field = field
        self.calc.electric_field = field

    def evaluate(self, atoms) -> dict:
        """Evaluate the requested properties for the current frame."""
        results = {
            "polarization": np.asarray(
                self.calc.get_property("polarization", atoms)
            ).reshape(3),
            "becs": np.asarray(self.calc.get_property("becs", atoms)).reshape(
                len(atoms), 9
            ),
            "polarizability": np.asarray(
                self.calc.get_property("polarizability", atoms)
            ).reshape(9),
        }

        if self.compute_energy:
            results.update(
                {
                    "energy": self.calc.get_property("energy", atoms),
                    "forces": self.calc.get_property("forces", atoms),
                    "stress": self.calc.get_property("stress", atoms),
                }
            )

        return results

    def clear_device_cache(self) -> None:
        """Release cached CUDA allocations between frames when possible."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_field_value(raw_value, default_field: np.ndarray) -> np.ndarray:
    """Extract a length-3 electric field vector from extxyz metadata."""
    if raw_value is None:
        return default_field.copy()

    if isinstance(raw_value, str):
        cleaned = raw_value.replace("[", " ").replace("]", " ").replace(",", " ")
        pieces = [piece for piece in cleaned.split() if piece]
        if len(pieces) == 3:
            return np.asarray([float(piece) for piece in pieces], dtype=float)

    field = np.asarray(raw_value, dtype=float).reshape(-1)
    if field.size != 3:
        raise ValueError(f"Expected a 3-vector electric field, got shape {field.shape}")
    return field


def _frame_field(atoms, default_field: np.ndarray) -> np.ndarray:
    """Choose the electric field for a frame based on existing metadata."""
    for key in ("MACE_electric_field", "REF_electric_field", "electric_field"):
        if key in atoms.info:
            return _parse_field_value(atoms.info[key], default_field)
    return default_field.copy()


def annotate_trajectory(
    input_path: Path,
    output_path: Path,
    calc: UnifiedMACEFieldCalculator,
    default_field: np.ndarray,
    overwrite: bool,
) -> None:
    """Read a trajectory, add MACEField properties, and write the result."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    first_frame = True
    for atoms in tqdm(iread(str(input_path), index=":"), desc="Annotating frames"):
        field = _frame_field(atoms, default_field)
        calc.electric_field = field
        results = calc.evaluate(atoms)

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
            str(output_path),
            atoms,
            format="extxyz",
            append=not first_frame,
        )
        calc.clear_device_cache()
        first_frame = False


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Backfill MACEField polarization/BEC/polarizability into an extxyz trajectory."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/home/brad/repositories/mace/MD_Tutorial/SiO2-mp-7000-210ps-2.xyz",
        help="Input extxyz trajectory to annotate in place.",
    )
    parser.add_argument(
        "--model-path",
        default=str(script_dir / "models" / "MACEField-omat-dielectric.model"),
        help="Path to the MACEField model file.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for unified inference, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--head-device",
        default=None,
        help="Optional explicit device for the selected head. Defaults to --device.",
    )
    parser.add_argument(
        "--polarization-device",
        default=None,
        help="Deprecated alias for --head-device.",
    )
    parser.add_argument(
        "--response-device",
        default=None,
        help="Deprecated alias for --head-device.",
    )
    parser.add_argument(
        "--energy-device",
        default=None,
        help="Deprecated alias for --head-device.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Inference dtype passed to MACECalculator.",
    )
    parser.add_argument(
        "--electric-field",
        default="0.0,0.0,0.0",
        help="Default electric field Ex,Ey,Ez used when the frame has no stored field metadata.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. By default the input file is replaced in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the explicit --output path if it already exists.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Unified head used for polarization, BECs, polarizability, and optional energy.",
    )
    parser.add_argument(
        "--energy-head",
        default=None,
        help="Deprecated alias for --head.",
    )
    parser.add_argument(
        "--polarization-head",
        default=None,
        help="Deprecated alias for --head.",
    )
    parser.add_argument(
        "--response-head",
        default=None,
        help="Deprecated alias for --head.",
    )
    parser.add_argument(
        "--compute-energy",
        action="store_true",
        help="Also recompute MACE_energy, MACE_stress, and MACE_forces. Disabled by default to reduce memory.",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Enable CuEq acceleration in MACECalculator when available.",
    )
    parser.add_argument(
        "--enable-oeq",
        action="store_true",
        help="Enable OEq acceleration in MACECalculator when available.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    legacy_heads = [
        value
        for value in (args.energy_head, args.polarization_head, args.response_head)
        if value
    ]
    unified_head = args.head or (legacy_heads[0] if legacy_heads else "pt_head")
    if any(value != unified_head for value in legacy_heads):
        raise ValueError(
            "Annotation is unified to a single head. Use --head to select one head."
        )

    legacy_devices = [
        value
        for value in (
            args.head_device,
            args.polarization_device,
            args.response_device,
            args.energy_device,
        )
        if value
    ]
    unified_device = args.head_device or args.device
    if legacy_devices and not args.head_device:
        unified_device = legacy_devices[0]
    if any(value != unified_device for value in legacy_devices):
        raise ValueError(
            "Annotation is unified to a single device. Use --device or --head-device."
        )

    default_field = _parse_field_value(args.electric_field, np.zeros(3, dtype=float))
    calc = UnifiedMACEFieldCalculator(
        model_path=args.model_path,
        device=unified_device,
        dtype=args.dtype,
        head=unified_head,
        enable_cueq=args.enable_cueq,
        enable_oeq=args.enable_oeq,
        compute_energy=args.compute_energy,
    )

    if args.output is None:
        tmp_output = input_path.with_name(input_path.name + ".tmp")
        if tmp_output.exists():
            tmp_output.unlink()
        annotate_trajectory(
            input_path=input_path,
            output_path=tmp_output,
            calc=calc,
            default_field=default_field,
            overwrite=True,
        )
        os.replace(tmp_output, input_path)
        print(f"Updated in place: {input_path}")
        return

    output_path = Path(args.output).expanduser().resolve()
    annotate_trajectory(
        input_path=input_path,
        output_path=output_path,
        calc=calc,
        default_field=default_field,
        overwrite=args.overwrite,
    )
    print(f"Wrote annotated trajectory: {output_path}")


if __name__ == "__main__":
    main()
