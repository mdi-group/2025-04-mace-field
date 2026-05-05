#!/usr/bin/env bash

# Fixed-cell SiO2 dielectric relaxation workflow inspired by the Allegro-pol
# quartz finite-field protocol. This script:
#   1. starts from the curated SiO2 preprocessed structure
#   2. builds an optional supercell while preserving the DFT lattice
#   3. relaxes atomic positions at zero field
#   4. relaxes atomic positions under small static electric fields along x/y/z
#   5. evaluates polarization and polarizability with the chosen MACEField model
#   6. writes compact CSV/JSON summaries for the directional finite-field runs
#
# Common overrides:
#   ./run_sio2_dielectric_relax.sh
#   MODEL_VARIANT=foundation ./run_sio2_dielectric_relax.sh
#   MACEFIELD_MODEL=/path/to/MACE-field-SiO2.model ./run_sio2_dielectric_relax.sh
#   SUPERCELL="1 1 1" FIELD_AMPLITUDE=0.03636 ./run_sio2_dielectric_relax.sh
#   FIELD_DIRECTIONS="z" FIELD_AMPLITUDE=0.03636 ./run_sio2_dielectric_relax.sh

set -eo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SIF:-$WORKDIR/macefield-lammps.sif}"
CONDA_SH="/home/brad/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-MACEField}"
MODELS_DIR="${MODELS_DIR:-$WORKDIR/models}"
LOGROOT="${LOGROOT:-$WORKDIR/logs}"
RUN_PARENT="${RUN_PARENT:-$WORKDIR/MD/runs}"
CREATE_LAMMPS_MODEL_SCRIPT="${CREATE_LAMMPS_MODEL_SCRIPT:-/home/brad/repositories/mace/mace-field/mace/cli/create_lammps_model.py}"

APPTAINER_BIN="${APPTAINER_BIN:-/usr/bin/apptainer}"
USE_SUDO="${USE_SUDO:-1}"
LAMMPS_ARGS="${LAMMPS_ARGS:--np 1}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
BACKGROUND_CHILD="${BACKGROUND_CHILD:-0}"
BACKGROUND_LOG_ONLY="${BACKGROUND_LOG_ONLY:-0}"

STRUCTURE_NAME="${STRUCTURE_NAME:-SiO2-mp-7000}"
PREPROCESSED_XYZ="${PREPROCESSED_XYZ:-/home/brad/repositories/2025-04-mace-field/scripts/SiO2/SiO2-preprocessed.xyz}"
SUPERCELL="${SUPERCELL:-1 1 1}"
RUN_TAG="${RUN_TAG:-}"
MODEL_VARIANT="${MODEL_VARIANT:-foundation}"

FIELD_AMPLITUDE="${FIELD_AMPLITUDE:-0.03636}"
FIELD_DIRECTIONS="${FIELD_DIRECTIONS:-x y z}"

MIN_STYLE="${MIN_STYLE:-cg/kk}"
MIN_ETOL="${MIN_ETOL:-0.0}"
MIN_FTOL="${MIN_FTOL:-1.0e-6}"
MIN_MAXITER="${MIN_MAXITER:-100}"
MIN_MAXEVAL="${MIN_MAXEVAL:-100}"
NEIGHBOR_SKIN="${NEIGHBOR_SKIN:-2.0}"
TIMESTEP_PS="${TIMESTEP_PS:-0.002}"

ANNOTATION_DEVICE="${ANNOTATION_DEVICE:-cuda}"
ANNOTATION_DTYPE="${ANNOTATION_DTYPE:-float32}"
ENABLE_CUEQ="${ENABLE_CUEQ:-0}"
ENABLE_OEQ="${ENABLE_OEQ:-0}"
MACEFIELD_HEAD="${MACEFIELD_HEAD:-}"
MACEFIELD_MODEL="${MACEFIELD_MODEL:-}"
LAMMPS_MODEL="${LAMMPS_MODEL:-}"
POSTPROCESS_MODEL="${POSTPROCESS_MODEL:-}"

DEFAULT_FOUNDATION_POSTPROCESS_MODEL="$WORKDIR/models/MACEField-omat-dielectric.model"
DEFAULT_FOUNDATION_LAMMPS_MODEL="$WORKDIR/models/MACEField-omat-dielectric.model-mliap_lammps.pt"
DEFAULT_FINETUNED_POSTPROCESS_MODEL="/home/brad/repositories/2025-04-mace-field/scripts/SiO2/MACE-field-SiO2.model"
DEFAULT_FINETUNED_LAMMPS_MODEL="/home/brad/repositories/2025-04-mace-field/scripts/SiO2/MACE-field-SiO2.model-mliap_lammps.pt"

RUN_LAMMPS_MODEL=""
RUN_POSTPROCESS_MODEL=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_model_variant() {
  case "${1,,}" in
    foundation|base|omat) printf 'foundation' ;;
    finetuned|fine_tuned|ft|sio2) printf 'finetuned' ;;
    *) return 1 ;;
  esac
}

default_head_for_variant() {
  case "$1" in
    foundation) printf 'pt_head' ;;
    finetuned) printf 'Default' ;;
    *) return 1 ;;
  esac
}

resolve_models() {
  if [[ -n "$LAMMPS_MODEL" && -n "$POSTPROCESS_MODEL" ]]; then
    return
  fi

  if [[ -n "$LAMMPS_MODEL" ]]; then
    POSTPROCESS_MODEL="${POSTPROCESS_MODEL:-${LAMMPS_MODEL%-mliap_lammps.pt}}"
    return
  fi

  if [[ -n "$POSTPROCESS_MODEL" ]]; then
    LAMMPS_MODEL="${LAMMPS_MODEL:-${POSTPROCESS_MODEL}-mliap_lammps.pt}"
    return
  fi

  if [[ "$MACEFIELD_MODEL" == *-mliap_lammps.pt ]]; then
    LAMMPS_MODEL="$MACEFIELD_MODEL"
    POSTPROCESS_MODEL="${MACEFIELD_MODEL%-mliap_lammps.pt}"
  else
    POSTPROCESS_MODEL="$MACEFIELD_MODEL"
    LAMMPS_MODEL="${MACEFIELD_MODEL}-mliap_lammps.pt"
  fi
}

apply_model_variant_defaults() {
  local variant="$1"
  local default_postprocess default_lammps

  case "$variant" in
    foundation)
      default_postprocess="$DEFAULT_FOUNDATION_POSTPROCESS_MODEL"
      default_lammps="$DEFAULT_FOUNDATION_LAMMPS_MODEL"
      ;;
    finetuned)
      default_postprocess="$DEFAULT_FINETUNED_POSTPROCESS_MODEL"
      default_lammps="$DEFAULT_FINETUNED_LAMMPS_MODEL"
      ;;
    *)
      die "Unknown MODEL_VARIANT: $variant"
      ;;
  esac

  if [[ -z "$POSTPROCESS_MODEL" && -z "$LAMMPS_MODEL" && -z "${MACEFIELD_MODEL:-}" ]]; then
    POSTPROCESS_MODEL="$default_postprocess"
    if [[ -f "$default_lammps" ]]; then
      LAMMPS_MODEL="$default_lammps"
    else
      LAMMPS_MODEL="$DEFAULT_FOUNDATION_LAMMPS_MODEL"
    fi
    return
  fi
}

cache_model_file() {
  local src="$1"
  local dest

  [[ -f "$src" ]] || die "Model file not found: $src"
  mkdir -p "$MODELS_DIR"

  dest="$MODELS_DIR/$(basename "$src")"
  if [[ "$src" != "$dest" ]]; then
    if [[ ! -f "$dest" || "$src" -nt "$dest" ]]; then
      cp -f "$src" "$dest"
    fi
  fi

  printf '%s\n' "$dest"
}

export_lammps_model() {
  local postprocess_model="$1"
  local lammps_model="$2"
  local head_arg=""

  [[ -f "$postprocess_model" ]] || die "Cannot export ML-IAP model; source model not found: $postprocess_model"
  [[ -f "$CREATE_LAMMPS_MODEL_SCRIPT" ]] || die "LAMMPS export helper not found: $CREATE_LAMMPS_MODEL_SCRIPT"
  [[ -f "$CONDA_SH" ]] || die "conda activation script not found: $CONDA_SH"

  if [[ -n "$MACEFIELD_HEAD" ]]; then
    head_arg="--head '$MACEFIELD_HEAD'"
  fi

  log "Generating ML-IAP export for $(basename "$postprocess_model") -> $(basename "$lammps_model")"
  bash -lc "set -eo pipefail; source '$CONDA_SH'; conda activate '$CONDA_ENV'; python '$CREATE_LAMMPS_MODEL_SCRIPT' '$postprocess_model' ${head_arg} --dtype float32 --format mliap"
  [[ -f "$lammps_model" ]] || die "Expected ML-IAP export was not created: $lammps_model"
}

prepare_model_artifacts() {
  if [[ -n "$POSTPROCESS_MODEL" ]]; then
    POSTPROCESS_MODEL="$(cache_model_file "$POSTPROCESS_MODEL")"
  fi

  if [[ -n "$LAMMPS_MODEL" && -f "$LAMMPS_MODEL" ]]; then
    LAMMPS_MODEL="$(cache_model_file "$LAMMPS_MODEL")"
  fi

  resolve_models

  if [[ -n "$POSTPROCESS_MODEL" ]]; then
    POSTPROCESS_MODEL="$(cache_model_file "$POSTPROCESS_MODEL")"
  fi

  if [[ -n "$LAMMPS_MODEL" ]]; then
    if [[ -f "$LAMMPS_MODEL" ]]; then
      LAMMPS_MODEL="$(cache_model_file "$LAMMPS_MODEL")"
    else
      LAMMPS_MODEL="$MODELS_DIR/$(basename "$LAMMPS_MODEL")"
    fi
  fi

  if [[ ! -f "$LAMMPS_MODEL" ]]; then
    export_lammps_model "$POSTPROCESS_MODEL" "$LAMMPS_MODEL"
  fi
}

stage_models_for_run() {
  local target_dir="$1"
  local lammps_name postprocess_name

  lammps_name="$(basename "$LAMMPS_MODEL")"
  postprocess_name="$(basename "$POSTPROCESS_MODEL")"

  cp -f "$LAMMPS_MODEL" "$target_dir/$lammps_name"
  cp -f "$POSTPROCESS_MODEL" "$target_dir/$postprocess_name"

  RUN_LAMMPS_MODEL="$target_dir/$lammps_name"
  RUN_POSTPROCESS_MODEL="$target_dir/$postprocess_name"
}

prepare_structure_from_xyz() {
  local source_xyz="$1"
  local target_dir="$2"
  local structure_xyz="$target_dir/${STRUCTURE_NAME}.xyz"
  local structure_data="$target_dir/structure.data"

  [[ -f "$source_xyz" ]] || die "Preprocessed structure not found: $source_xyz"
  mkdir -p "$target_dir"

  python - <<PY
from pathlib import Path
from ase.io import read, write

source = Path(${source_xyz@Q})
target_dir = Path(${target_dir@Q})
structure_xyz = Path(${structure_xyz@Q})
structure_data = Path(${structure_data@Q})
supercell = tuple(int(x) for x in ${SUPERCELL@Q}.split())

atoms = read(source)
atoms = atoms.repeat(supercell)
atoms.pbc = True

write(structure_xyz, atoms, format="extxyz")
write(
    structure_data,
    atoms,
    format="lammps-data",
    specorder=["O", "Si"],
    atom_style="atomic",
)

print(f"Name:        ${STRUCTURE_NAME}")
print(f"Directory:   {target_dir}")
print("Elements:    O, Si  (LAMMPS types 1..N in this order)")
print(f"LAMMPS data:  {structure_data}")
print(f"XYZ:         {structure_xyz}")
print("Infile:      (not written)")
PY
}

run_lammps_input() {
  local infile="$1"
  local stage="$2"
  local field_x="$3"
  local field_y="$4"
  local field_z="$5"
  local -a apptainer_cmd env_args lmp_args

  read -r -a lmp_args <<< "$LAMMPS_ARGS"

  if is_true "$USE_SUDO"; then
    apptainer_cmd=(sudo "$APPTAINER_BIN")
  else
    apptainer_cmd=("$APPTAINER_BIN")
  fi

  env_args=(
    --env "MACE_EFIELD_MODE=env"
    --env "MACE_EFIELD=${field_x},${field_y},${field_z}"
    --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  )

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    env_args+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
  fi

  log "Launching ${stage}: ${infile} with field (${field_x}, ${field_y}, ${field_z})"
  "${apptainer_cmd[@]}" exec --cleanenv --nv --pwd "$(pwd)" "${env_args[@]}" "$SIF" \
    lmp_mpi "${lmp_args[@]}" -in "$infile"
}

read -r -a SUPERCELL_ARR <<< "$SUPERCELL"
[[ ${#SUPERCELL_ARR[@]} -eq 3 ]] || die "SUPERCELL must contain exactly three integers, e.g. SUPERCELL=\"1 1 1\""
read -r -a FIELD_DIR_ARR <<< "$FIELD_DIRECTIONS"
[[ ${#FIELD_DIR_ARR[@]} -ge 1 ]] || die "FIELD_DIRECTIONS must contain at least one of x y z"

MODEL_VARIANT_NORMALIZED="$(normalize_model_variant "$MODEL_VARIANT")" || die "MODEL_VARIANT must be one of: foundation, finetuned"
if [[ -z "$MACEFIELD_HEAD" ]]; then
  MACEFIELD_HEAD="$(default_head_for_variant "$MODEL_VARIANT_NORMALIZED")" || die "Could not determine default MACEFIELD_HEAD"
fi

SUPER_TAG="${SUPERCELL_ARR[0]}x${SUPERCELL_ARR[1]}x${SUPERCELL_ARR[2]}"
if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="${STRUCTURE_NAME}-sc${SUPER_TAG}-dielectric-relax-$(date '+%F_%H%M%S')"
fi

RUN_DIR="$RUN_PARENT/$RUN_TAG"
STRUCTURE_DIR="$RUN_DIR/$STRUCTURE_NAME"
LOGFILE="$LOGROOT/${RUN_TAG}.log"

mkdir -p "$RUN_PARENT" "$LOGROOT"

if is_true "$RUN_IN_BACKGROUND" && ! is_true "$BACKGROUND_CHILD"; then
  env RUN_TAG="$RUN_TAG" BACKGROUND_CHILD=1 BACKGROUND_LOG_ONLY=1 bash "$0" "$@" >"$LOGFILE" 2>&1 &
  printf 'Started background run: %s\n' "$RUN_TAG"
  printf 'PID: %s\n' "$!"
  printf 'Log: %s\n' "$LOGFILE"
  exit 0
fi

[[ -f "$SIF" ]] || die "Container not found: $SIF"
[[ -f "$CONDA_SH" ]] || die "conda activation script not found: $CONDA_SH"

apply_model_variant_defaults "$MODEL_VARIANT_NORMALIZED"
prepare_model_artifacts
[[ -f "$LAMMPS_MODEL" ]] || die "LAMMPS MACEField model not found: $LAMMPS_MODEL"
[[ -f "$POSTPROCESS_MODEL" ]] || die "Python MACEField model not found: $POSTPROCESS_MODEL"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

set -euo pipefail
if is_true "$BACKGROUND_LOG_ONLY"; then
  exec >>"$LOGFILE" 2>&1
else
  exec > >(tee -a "$LOGFILE") 2>&1
fi

log "Workflow root: $RUN_DIR"
log "Model variant: $MODEL_VARIANT_NORMALIZED"
if is_true "$BACKGROUND_CHILD"; then
  log "Background mode: PID $$"
fi
log "Preprocessed start: $PREPROCESSED_XYZ"
log "Supercell: ${SUPERCELL_ARR[*]}"
log "LAMMPS model: $LAMMPS_MODEL"
log "Postprocess model: $POSTPROCESS_MODEL"
log "Head: $MACEFIELD_HEAD"
log "Field amplitude (V/Ang): ${FIELD_AMPLITUDE}"
log "Field directions: ${FIELD_DIRECTIONS}"
log "Relaxation protocol: zero-field fixed-cell minimize -> finite-field fixed-cell minimizes along requested directions"

prepare_structure_from_xyz "$PREPROCESSED_XYZ" "$STRUCTURE_DIR"
cd "$STRUCTURE_DIR"
stage_models_for_run "$STRUCTURE_DIR"
log "Staged LAMMPS model: $RUN_LAMMPS_MODEL"
log "Staged postprocess model: $RUN_POSTPROCESS_MODEL"

ZERO_INPUT="in.01_relax_zero_field"
ZERO_DATA="relaxed_zero.data"
ZERO_EXTXYZ="relaxed_zero.annotated.extxyz"
SUMMARY_CSV="dielectric_relax_summary.csv"
SUMMARY_JSON="dielectric_relax_summary.json"

cat >"$ZERO_INPUT" <<EOF
# --- 01: fixed-cell SiO2 zero-field relaxation ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_data       structure.data

mass 1 15.999
mass 2 28.085

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Si

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

timestep        ${TIMESTEP_PS}

thermo          1
thermo_style    custom step pe fmax fnorm press pxx pyy pzz pxy pxz pyz vol lx ly lz
thermo_modify   flush yes

dump            relax_zero all custom 1 relax_zero.lammpstrj id type xu yu zu
dump_modify     relax_zero sort id

min_style       ${MIN_STYLE}
minimize        ${MIN_ETOL} ${MIN_FTOL} ${MIN_MAXITER} ${MIN_MAXEVAL}

undump          relax_zero
write_data      ${ZERO_DATA}
EOF

run_lammps_input "$ZERO_INPUT" "zero-field relaxation" "0.0" "0.0" "0.0"

FIELD_DATA_FILES=()
FIELD_EXTXYZ_FILES=()
FIELD_DIRECTION_NAMES=()

for direction in "${FIELD_DIR_ARR[@]}"; do
  case "${direction,,}" in
    x|a)
      field_axis="x"
      field_x="$FIELD_AMPLITUDE"
      field_y="0.0"
      field_z="0.0"
      ;;
    y|b)
      field_axis="y"
      field_x="0.0"
      field_y="$FIELD_AMPLITUDE"
      field_z="0.0"
      ;;
    z|c)
      field_axis="z"
      field_x="0.0"
      field_y="0.0"
      field_z="$FIELD_AMPLITUDE"
      ;;
    *)
      die "Unsupported field direction: $direction (use x, y, or z)"
      ;;
  esac

  FIELD_INPUT="in.02_relax_finite_field_${field_axis}"
  FIELD_DATA="relaxed_field_${field_axis}.data"
  FIELD_EXTXYZ="relaxed_field_${field_axis}.annotated.extxyz"

  cat >"$FIELD_INPUT" <<EOF
# --- 02: fixed-cell SiO2 finite-field relaxation (${field_axis}) ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_data       ${ZERO_DATA}

mass 1 15.999
mass 2 28.085

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Si

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

timestep        ${TIMESTEP_PS}

thermo          1
thermo_style    custom step pe fmax fnorm press pxx pyy pzz pxy pxz pyz vol lx ly lz
thermo_modify   flush yes

dump            relax_field all custom 1 relaxed_field_${field_axis}.lammpstrj id type xu yu zu
dump_modify     relax_field sort id

min_style       ${MIN_STYLE}
minimize        ${MIN_ETOL} ${MIN_FTOL} ${MIN_MAXITER} ${MIN_MAXEVAL}

undump          relax_field
write_data      ${FIELD_DATA}
EOF

  run_lammps_input "$FIELD_INPUT" "finite-field relaxation (${field_axis})" "$field_x" "$field_y" "$field_z"
  FIELD_DATA_FILES+=("$FIELD_DATA")
  FIELD_EXTXYZ_FILES+=("$FIELD_EXTXYZ")
  FIELD_DIRECTION_NAMES+=("$field_axis")
done

FIELD_DIRECTIONS_STR="$(IFS=' '; printf '%s' "${FIELD_DIRECTION_NAMES[*]}")"
FIELD_DATA_STR="$(IFS=' '; printf '%s' "${FIELD_DATA_FILES[*]}")"
FIELD_EXTXYZ_STR="$(IFS=' '; printf '%s' "${FIELD_EXTXYZ_FILES[*]}")"

python - <<PY
import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

run_postprocess_model = Path(${RUN_POSTPROCESS_MODEL@Q})
zero_data = Path(${ZERO_DATA@Q})
zero_extxyz = Path(${ZERO_EXTXYZ@Q})
summary_csv = Path(${SUMMARY_CSV@Q})
summary_json = Path(${SUMMARY_JSON@Q})
head = ${MACEFIELD_HEAD@Q}
device = ${ANNOTATION_DEVICE@Q}
dtype = ${ANNOTATION_DTYPE@Q}
enable_cueq = ${ENABLE_CUEQ@Q}.lower() in {"1", "true", "yes", "on"}
enable_oeq = ${ENABLE_OEQ@Q}.lower() in {"1", "true", "yes", "on"}
field_amplitude = float(${FIELD_AMPLITUDE@Q})
field_directions = ${FIELD_DIRECTIONS_STR@Q}.split()
field_data_files = ${FIELD_DATA_STR@Q}.split()
field_extxyz_files = ${FIELD_EXTXYZ_STR@Q}.split()
eps0_e_per_vang = 0.005526349406

def read_lammps_atomic(path: Path):
    atoms = read(
        path,
        format="lammps-data",
        style="atomic",
        Z_of_type={1: 8, 2: 14},
    )
    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.cell.copy(),
        pbc=atoms.pbc.copy(),
    )

def build_calc():
    return MACECalculator(
        model_paths=str(run_postprocess_model),
        device=device,
        default_dtype=dtype,
        model_type="MACEField",
        head=head,
        enable_cueq=enable_cueq,
        enable_oeq=enable_oeq,
    )

def evaluate(atoms, electric_field):
    calc = build_calc()
    calc.electric_field = np.asarray(electric_field, dtype=float).reshape(3)
    atoms.calc = calc
    polarization = np.asarray(calc.get_property("polarization", atoms), dtype=float).reshape(3)
    polarizability = np.asarray(calc.get_property("polarizability", atoms), dtype=float).reshape(3, 3)
    energy = float(calc.get_property("energy", atoms))
    forces = np.asarray(calc.get_property("forces", atoms), dtype=float)
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
    atoms.info["MACE_electric_field"] = np.asarray(electric_field, dtype=float).reshape(3)
    atoms.info["MACE_polarization"] = polarization
    atoms.info["MACE_polarizability"] = polarizability.reshape(9)
    atoms.info["MACE_energy"] = energy
    return {
        "polarization": polarization,
        "polarizability": polarizability,
        "energy": energy,
        "max_force": max_force,
    }

zero_atoms = read_lammps_atomic(zero_data)
zero_atoms.pbc = True
zero_eval = evaluate(zero_atoms, np.zeros(3))
write(zero_extxyz, zero_atoms, format="extxyz")

field_results = {}
for direction, data_file, extxyz_file in zip(field_directions, field_data_files, field_extxyz_files):
    axis_index = {"x": 0, "y": 1, "z": 2}[direction]
    field_vec = np.zeros(3, dtype=float)
    field_vec[axis_index] = field_amplitude

    field_atoms = read_lammps_atomic(Path(data_file))
    field_atoms.pbc = True
    field_eval = evaluate(field_atoms, field_vec)
    write(Path(extxyz_file), field_atoms, format="extxyz")

    delta_p = field_eval["polarization"] - zero_eval["polarization"]
    eps_diag = np.full(3, np.nan, dtype=float)
    eps_diag[axis_index] = 1.0 + delta_p[axis_index] / (eps0_e_per_vang * field_amplitude)

    field_results[direction] = {
        "field_vector": field_vec.tolist(),
        "field_MV_per_cm": (field_vec * 100.0).tolist(),
        "energy_eV": field_eval["energy"],
        "max_force_eV_per_A": field_eval["max_force"],
        "polarization_e_per_A2": field_eval["polarization"].tolist(),
        "polarization_uC_per_cm2": (field_eval["polarization"] * 1602.176634).tolist(),
        "polarizability_tensor": field_eval["polarizability"].tolist(),
        "delta_polarization_e_per_A2": delta_p.tolist(),
        "delta_polarization_uC_per_cm2": (delta_p * 1602.176634).tolist(),
        "dielectric_estimate_diagonal": {
            "xx": None if np.isnan(eps_diag[0]) else float(eps_diag[0]),
            "yy": None if np.isnan(eps_diag[1]) else float(eps_diag[1]),
            "zz": None if np.isnan(eps_diag[2]) else float(eps_diag[2]),
        },
    }

summary = {
    "model_path": str(run_postprocess_model),
    "head": head,
    "field_amplitude_V_per_A": field_amplitude,
    "field_amplitude_MV_per_cm": field_amplitude * 100.0,
    "field_directions": field_directions,
    "eps0_e_per_VA": eps0_e_per_vang,
    "zero_field": {
        "energy_eV": zero_eval["energy"],
        "max_force_eV_per_A": zero_eval["max_force"],
        "polarization_e_per_A2": zero_eval["polarization"].tolist(),
        "polarization_uC_per_cm2": (zero_eval["polarization"] * 1602.176634).tolist(),
        "polarizability_tensor": zero_eval["polarizability"].tolist(),
    },
    "finite_field_runs": field_results,
}

with summary_json.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

fieldnames = [
    "direction",
    "component",
    "field_V_per_A",
    "field_MV_per_cm",
    "P0_e_per_A2",
    "Pfield_e_per_A2",
    "deltaP_e_per_A2",
    "P0_uC_per_cm2",
    "Pfield_uC_per_cm2",
    "deltaP_uC_per_cm2",
    "dielectric_estimate",
]

key_map = {"x": "xx", "y": "yy", "z": "zz"}

with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for direction in field_directions:
        axis_index = {"x": 0, "y": 1, "z": 2}[direction]
        result = field_results[direction]
        writer.writerow(
            {
                "direction": direction,
                "component": direction,
                "field_V_per_A": field_amplitude,
                "field_MV_per_cm": field_amplitude * 100.0,
                "P0_e_per_A2": zero_eval["polarization"][axis_index],
                "Pfield_e_per_A2": result["polarization_e_per_A2"][axis_index],
                "deltaP_e_per_A2": result["delta_polarization_e_per_A2"][axis_index],
                "P0_uC_per_cm2": zero_eval["polarization"][axis_index] * 1602.176634,
                "Pfield_uC_per_cm2": result["polarization_uC_per_cm2"][axis_index],
                "deltaP_uC_per_cm2": result["delta_polarization_uC_per_cm2"][axis_index],
                "dielectric_estimate": result["dielectric_estimate_diagonal"][key_map[direction]],
            }
        )
PY

log "Finished."
log "Structure directory: $STRUCTURE_DIR"
log "Zero-field relaxed structure: $STRUCTURE_DIR/$ZERO_EXTXYZ"
for extxyz_file in "${FIELD_EXTXYZ_FILES[@]}"; do
  log "Finite-field relaxed structure: $STRUCTURE_DIR/$extxyz_file"
done
log "Summary CSV: $STRUCTURE_DIR/$SUMMARY_CSV"
log "Summary JSON: $STRUCTURE_DIR/$SUMMARY_JSON"
log "Log file: $LOGFILE"
