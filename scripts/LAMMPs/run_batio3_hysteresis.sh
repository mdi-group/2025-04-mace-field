#!/usr/bin/env bash

# Workflow launcher for BaTiO3 (mp-5986) finite-field hysteresis MD with MACEField.
#
# What it does:
#   1. starts from a curated BaTiO3 structure (or optionally fetches from MP)
#   2. builds a configurable supercell while preserving the input lattice
#   3. relaxes atomic positions at fixed cell, then equilibrates the supercell with MACEField in LAMMPS
#   4. runs a finite-field driven trajectory with a sinusoidal field along z/c
#   5. logs response properties live during the driven run via fix python/invoke
#   6. optionally converts the LAMMPS dump + thermo sidecar into extxyz
#   7. optionally backfills response properties after the run as a fallback
#
# Common overrides:
#   SUPERCELL="1 1 1" LAMMPS_ARGS="-np 4" ./run_batio3_hysteresis.sh
#   STRUCTURE_SOURCE=mp MP_API_KEY=... ./run_batio3_hysteresis.sh
#   TEMPERATURE_K=300 FIELD_MAX_Z=0.3636 FIELD_FREQUENCY_GHZ=5.0 FIELD_CYCLES=2 ./run_batio3_hysteresis.sh
#   TEMPERATURE_K=0 FIELD_MAX_Z=0.3636 ./run_batio3_hysteresis.sh
#   MACEFIELD_MODEL=models/MACEField-omat-dielectric.model ./run_batio3_hysteresis.sh
#   MODEL_VARIANT=finetuned ./run_batio3_hysteresis.sh
#   START_STAGE=02 RUN_TAG=BaTiO3-mp-5986-sc1x1x1-0K-5GHz-hysteresis-YYYY-MM-DD_HHMMSS ./run_batio3_hysteresis.sh

set -eo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SIF:-$WORKDIR/macefield-lammps.sif}"
CONDA_SH="/home/brad/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-MACEField}"
LOGROOT="${LOGROOT:-$WORKDIR/logs}"

MPID="${MPID:-mp-5986}"
STRUCTURE_NAME="${STRUCTURE_NAME:-BaTiO3-mp-5986}"
STRUCTURE_SOURCE="${STRUCTURE_SOURCE:-preprocessed}"
PREPROCESSED_XYZ="${PREPROCESSED_XYZ:-/home/brad/repositories/2025-04-mace-field/scripts/BaTiO3/BaTiO3-preprocessed.xyz}"
SUPERCELL="${SUPERCELL:-1 1 1}"
RUN_PARENT="${RUN_PARENT:-$WORKDIR/MD/runs}"
RUN_TAG="${RUN_TAG:-}"
START_STAGE="${START_STAGE:-all}"
MODEL_VARIANT="${MODEL_VARIANT:-foundation}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
BACKGROUND_CHILD="${BACKGROUND_CHILD:-0}"
BACKGROUND_LOG_ONLY="${BACKGROUND_LOG_ONLY:-0}"

TEMPERATURE_K="${TEMPERATURE_K:-0.0}"
TIMESTEP_PS="${TIMESTEP_PS:-0.002}"
EQUIL_STEPS="${EQUIL_STEPS:-20000}"
TDAMP_PS="${TDAMP_PS:-0.1}"
PDAMP_PS="${PDAMP_PS:-1.0}"
NEIGHBOR_SKIN="${NEIGHBOR_SKIN:-2.0}"
VELOCITY_SEED="${VELOCITY_SEED:-4928459}"
MIN_STYLE="${MIN_STYLE:-cg/kk}"
MIN_ETOL="${MIN_ETOL:-1.0e-10}"
MIN_FTOL="${MIN_FTOL:-1.0e-10}"
MIN_MAXITER="${MIN_MAXITER:-20000}"
MIN_MAXEVAL="${MIN_MAXEVAL:-200000}"
BOX_RELAX_VMAX="${BOX_RELAX_VMAX:-0.001}"
ATHERMAL_EQUIL_TIME_PS="${ATHERMAL_EQUIL_TIME_PS:-10.0}"
ATHERMAL_VISCOUS_GAMMA="${ATHERMAL_VISCOUS_GAMMA:-0.5}"
RELAX_POSITIONS_BEFORE_MD="${RELAX_POSITIONS_BEFORE_MD:-1}"

EQ_DUMP_EVERY="${EQ_DUMP_EVERY:-100}"
EQ_THERMO_EVERY="${EQ_THERMO_EVERY:-100}"
DUMP_EVERY="${DUMP_EVERY:-1}"
THERMO_EVERY="${THERMO_EVERY:-$DUMP_EVERY}"

FIELD_MAX_Z="${FIELD_MAX_Z:-0.3636}"
FIELD_FREQUENCY_GHZ="${FIELD_FREQUENCY_GHZ:-5.0}"
FIELD_CYCLES="${FIELD_CYCLES:-1}"
FIELD_PHASE_RAD="${FIELD_PHASE_RAD:-0.0}"

EX_FIELD="${EX_FIELD:-0.0}"
EY_FIELD="${EY_FIELD:-0.0}"
EZ_FIELD="${EZ_FIELD:-0.0}"

ANNOTATE_TRAJECTORY="${ANNOTATE_TRAJECTORY:-1}"
ANNOTATE_COMPUTE_ENERGY="${ANNOTATE_COMPUTE_ENERGY:-0}"
ANNOTATION_DEVICE="${ANNOTATION_DEVICE:-auto}"
ANNOTATION_DTYPE="${ANNOTATION_DTYPE:-float32}"
ANNOTATION_GPU_LOCK_ROOT="${ANNOTATION_GPU_LOCK_ROOT:-/tmp/macefield_annotation_gpu_locks}"
MACEFIELD_HEAD="${MACEFIELD_HEAD:-}"
POLARIZATION_DEVICE="${POLARIZATION_DEVICE:-}"
RESPONSE_DEVICE="${RESPONSE_DEVICE:-}"
ENERGY_DEVICE="${ENERGY_DEVICE:-}"
ENABLE_CUEQ="${ENABLE_CUEQ:-0}"
ENABLE_OEQ="${ENABLE_OEQ:-0}"

LIVE_RESPONSE_LOGGING="${LIVE_RESPONSE_LOGGING:-0}"
LIVE_RESPONSE_EVERY="${LIVE_RESPONSE_EVERY:-$DUMP_EVERY}"
LIVE_RESPONSE_EXTXYZ="${LIVE_RESPONSE_EXTXYZ:-hysteresis.live.extxyz}"
LIVE_RESPONSE_TSV="${LIVE_RESPONSE_TSV:-hysteresis.live.tsv}"
LIVE_LOGGER_DEVICE="${LIVE_LOGGER_DEVICE:-cuda}"
LIVE_LOGGER_DTYPE="${LIVE_LOGGER_DTYPE:-float32}"
LIVE_COMPUTE_ENERGY="${LIVE_COMPUTE_ENERGY:-0}"
LIVE_POLARIZATION_DEVICE="${LIVE_POLARIZATION_DEVICE:-}"
LIVE_RESPONSE_DEVICE="${LIVE_RESPONSE_DEVICE:-}"
LIVE_ENERGY_DEVICE="${LIVE_ENERGY_DEVICE:-}"
LIVE_ENABLE_CUEQ="${LIVE_ENABLE_CUEQ:-$ENABLE_CUEQ}"
LIVE_ENABLE_OEQ="${LIVE_ENABLE_OEQ:-$ENABLE_OEQ}"

MACEFIELD_MODEL="${MACEFIELD_MODEL:-}"
LAMMPS_MODEL="${LAMMPS_MODEL:-}"
POSTPROCESS_MODEL="${POSTPROCESS_MODEL:-}"
RUN_LAMMPS_MODEL=""
RUN_POSTPROCESS_MODEL=""
RUN_LIVE_LOGGER=""
MODELS_DIR="${MODELS_DIR:-$WORKDIR/models}"
CREATE_LAMMPS_MODEL_SCRIPT="${CREATE_LAMMPS_MODEL_SCRIPT:-/home/brad/repositories/mace/mace-field/mace/cli/create_lammps_model.py}"

DEFAULT_FOUNDATION_POSTPROCESS_MODEL="$WORKDIR/models/MACEField-omat-dielectric.model"
DEFAULT_FOUNDATION_LAMMPS_MODEL="$WORKDIR/models/MACEField-omat-dielectric.model-mliap_lammps.pt"
DEFAULT_FINETUNED_POSTPROCESS_MODEL="/home/brad/repositories/2025-04-mace-field/scripts/BaTiO3/MACE-Field-BaTiO3.model"
DEFAULT_FINETUNED_LAMMPS_MODEL="/home/brad/repositories/2025-04-mace-field/scripts/BaTiO3/MACE-Field-BaTiO3.model-mliap_lammps.pt"

APPTAINER_BIN="${APPTAINER_BIN:-/usr/bin/apptainer}"
USE_SUDO="${USE_SUDO:-1}"
LAMMPS_ARGS="${LAMMPS_ARGS:--np 4}"

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

ANNOTATION_GPU_LOCK=""

release_annotation_gpu_lock() {
  if [[ -n "${ANNOTATION_GPU_LOCK:-}" && -e "${ANNOTATION_GPU_LOCK:-}" ]]; then
    rm -f "$ANNOTATION_GPU_LOCK"
  fi
}

visible_gpu_ids() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | awk 'NF'
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF'
    return
  fi

  return 1
}

select_annotation_device() {
  local requested="${ANNOTATION_DEVICE:-auto}"
  local requested_lc="${requested,,}"
  local best_gpu="" best_count="" gpu count pid

  case "$requested_lc" in
    auto|cuda)
      ;;
    cpu|cpu:*|cuda:*|mps|mps:*|xpu|xpu:*)
      printf '%s\n' "$requested"
      return
      ;;
    *)
      printf '%s\n' "$requested"
      return
      ;;
  esac

  mkdir -p "$ANNOTATION_GPU_LOCK_ROOT"

  while IFS= read -r gpu; do
    [[ -n "$gpu" ]] || continue
    count=0
    for lock in "$ANNOTATION_GPU_LOCK_ROOT"/gpu"${gpu}".pid*; do
      [[ -e "$lock" ]] || continue
      pid="${lock##*.pid}"
      if kill -0 "$pid" 2>/dev/null; then
        count=$((count + 1))
      else
        rm -f "$lock"
      fi
    done

    if [[ -z "$best_gpu" || "$count" -lt "$best_count" ]]; then
      best_gpu="$gpu"
      best_count="$count"
    fi
  done < <(visible_gpu_ids || true)

  if [[ -n "$best_gpu" ]]; then
    ANNOTATION_GPU_LOCK="$ANNOTATION_GPU_LOCK_ROOT/gpu${best_gpu}.pid$$"
    : >"$ANNOTATION_GPU_LOCK"
    trap release_annotation_gpu_lock EXIT
    printf 'cuda:%s\n' "$best_gpu"
    return
  fi

  if [[ "$requested_lc" == "cuda" ]]; then
    printf 'cuda\n'
  else
    printf 'cpu\n'
  fi
}

normalize_stage() {
  case "${1,,}" in
    all|full|01|1) printf 'all' ;;
    02|2|drive|production|resume) printf '02' ;;
    *) return 1 ;;
  esac
}

normalize_model_variant() {
  case "${1,,}" in
    foundation|base|omat) printf 'foundation' ;;
    finetuned|fine_tuned|ft|batio3) printf 'finetuned' ;;
    *) return 1 ;;
  esac
}

normalize_structure_source() {
  case "${1,,}" in
    preprocessed|local|dft) printf 'preprocessed' ;;
    mp|materialsproject|materials_project|fetch) printf 'mp' ;;
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

  if [[ -z "$POSTPROCESS_MODEL" && -n "$LAMMPS_MODEL" ]]; then
    return
  fi

  if [[ -z "$LAMMPS_MODEL" && -n "$POSTPROCESS_MODEL" ]]; then
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

stage_runtime_helpers() {
  local target_dir="$1"
  local live_logger_name

  live_logger_name="$(basename "$WORKDIR/live_macefield_logger.py")"
  cp -f "$WORKDIR/live_macefield_logger.py" "$target_dir/$live_logger_name"
  RUN_LIVE_LOGGER="$target_dir/$live_logger_name"
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
    specorder=["O", "Ti", "Ba"],
    atom_style="atomic",
)

print(f"Name:        ${STRUCTURE_NAME}")
print(f"Directory:   {target_dir}")
print("Elements:    O, Ti, Ba  (LAMMPS types 1..N in this order)")
print(f"LAMMPS data:  {structure_data}")
print(f"XYZ:         {structure_xyz}")
print("Infile:      (not written)")
PY
}

run_lammps_input() {
  local infile="$1"
  local stage="$2"
  local field_x="${3:-$EX_FIELD}"
  local field_y="${4:-$EY_FIELD}"
  local field_z="${5:-$EZ_FIELD}"
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

  log "Launching ${stage}: ${infile}"
  "${apptainer_cmd[@]}" exec --cleanenv --nv --pwd "$(pwd)" "${env_args[@]}" "$SIF" \
    lmp_mpi "${lmp_args[@]}" -in "$infile"
}

read -r -a SUPERCELL_ARR <<< "$SUPERCELL"
[[ ${#SUPERCELL_ARR[@]} -eq 3 ]] || die "SUPERCELL must contain exactly three integers, e.g. SUPERCELL=\"2 2 2\""

START_STAGE_NORMALIZED="$(normalize_stage "$START_STAGE")" || die "START_STAGE must be one of: all, 01, 02, drive, production, resume"
MODEL_VARIANT_NORMALIZED="$(normalize_model_variant "$MODEL_VARIANT")" || die "MODEL_VARIANT must be one of: foundation, finetuned"
STRUCTURE_SOURCE_NORMALIZED="$(normalize_structure_source "$STRUCTURE_SOURCE")" || die "STRUCTURE_SOURCE must be one of: preprocessed, mp"
if [[ -z "$MACEFIELD_HEAD" ]]; then
  MACEFIELD_HEAD="$(default_head_for_variant "$MODEL_VARIANT_NORMALIZED")" || die "Could not determine default MACEFIELD_HEAD"
fi

SUPER_TAG="${SUPERCELL_ARR[0]}x${SUPERCELL_ARR[1]}x${SUPERCELL_ARR[2]}"
TEMP_TAG="$(python - <<PY
temp_k = float("${TEMPERATURE_K}")
if abs(temp_k - round(temp_k)) < 1.0e-9:
    print(f"{int(round(temp_k))}K")
else:
    print(f"{temp_k:g}K")
PY
)"
if [[ -z "$RUN_TAG" ]]; then
  [[ "$START_STAGE_NORMALIZED" == "all" ]] || die "RUN_TAG is required when START_STAGE=$START_STAGE"
  RUN_TAG="${STRUCTURE_NAME}-sc${SUPER_TAG}-${TEMP_TAG}-5GHz-hysteresis-$(date '+%F_%H%M%S')"
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
if [[ "$STRUCTURE_SOURCE_NORMALIZED" == "mp" ]]; then
  [[ -n "${MP_API_KEY:-}" ]] || die "MP_API_KEY is not set (required when STRUCTURE_SOURCE=mp)"
fi

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

FIELD_PERIOD_PS="$(python - <<PY
freq_ghz = float("${FIELD_FREQUENCY_GHZ}")
if freq_ghz <= 0.0:
    raise SystemExit("FIELD_FREQUENCY_GHZ must be positive")
print(f"{1000.0 / freq_ghz:.12g}")
PY
)"
FIELD_PERIOD_STEPS="$(python - <<PY
period_ps = float("${FIELD_PERIOD_PS}")
timestep_ps = float("${TIMESTEP_PS}")
steps = round(period_ps / timestep_ps)
if steps <= 0:
    raise SystemExit("Computed field period in steps must be positive")
print(int(steps))
PY
)"
DRIVE_STEPS="$(python - <<PY
cycles = float("${FIELD_CYCLES}")
period_steps = int("${FIELD_PERIOD_STEPS}")
steps = round(cycles * period_steps)
if steps <= 0:
    raise SystemExit("Computed driven-run steps must be positive")
print(int(steps))
PY
)"
DRIVE_TIME_PS="$(python - <<PY
steps = int("${DRIVE_STEPS}")
timestep_ps = float("${TIMESTEP_PS}")
print(f"{steps * timestep_ps:.12g}")
PY
)"
DRIVE_START_EZ="$(python - <<PY
import math
e0 = float("${FIELD_MAX_Z}")
phase = float("${FIELD_PHASE_RAD}")
print(f"{e0 * math.cos(phase):.12g}")
PY
)"
IS_ZERO_TEMP="$(python - <<PY
temp_k = float("${TEMPERATURE_K}")
print("1" if abs(temp_k) < 1.0e-12 else "0")
PY
)"

log "Workflow root: $RUN_DIR"
log "Start stage: $START_STAGE_NORMALIZED"
log "Structure source: $STRUCTURE_SOURCE_NORMALIZED"
if [[ "$STRUCTURE_SOURCE_NORMALIZED" == "preprocessed" ]]; then
  log "Preprocessed start: $PREPROCESSED_XYZ"
fi
log "Model variant: $MODEL_VARIANT_NORMALIZED"
if is_true "$BACKGROUND_CHILD"; then
  log "Background mode: PID $$"
fi
log "Supercell: ${SUPERCELL_ARR[*]}"
log "LAMMPS model: $LAMMPS_MODEL"
log "Postprocess model: $POSTPROCESS_MODEL"
log "Target temperature: ${TEMPERATURE_K} K"
log "Field drive: Ez(t) = ${FIELD_MAX_Z} * cos(2*pi*t/${FIELD_PERIOD_PS} ps + ${FIELD_PHASE_RAD})"
log "Field frequency: ${FIELD_FREQUENCY_GHZ} GHz"
log "Field period: ${FIELD_PERIOD_PS} ps (${FIELD_PERIOD_STEPS} steps)"
log "Driven run: ${FIELD_CYCLES} cycle(s), ${DRIVE_TIME_PS} ps, ${DRIVE_STEPS} steps"
log "Equilibration field: Ex=0 Ey=0 Ez=${DRIVE_START_EZ}"
if [[ "${SUPERCELL_ARR[*]}" != "3 3 3" ]]; then
  log "Using a reduced or expanded BaTiO3 supercell; Allegro-pol 0 K hysteresis used 3 3 3 (135 atoms)."
fi
log "Production sampling: every ${DUMP_EVERY} step(s)"
if is_true "$LIVE_RESPONSE_LOGGING"; then
  log "Live response logging: every ${LIVE_RESPONSE_EVERY} step(s) -> ${LIVE_RESPONSE_EXTXYZ}"
fi
if [[ "$IS_ZERO_TEMP" == "1" ]]; then
  log "Athermal mode: using NVE + viscous damping for ${ATHERMAL_EQUIL_TIME_PS} ps under Emax, then driven NVE + viscous damping."
  log "Paper note: Allegro-pol's main-text 0 K hysteresis used structural relaxations on a 135-atom (3 3 3) cell; this script uses a damped-MD proxy."
fi

ATHERMAL_EQUIL_STEPS=""
if [[ "$IS_ZERO_TEMP" == "1" ]]; then
  ATHERMAL_EQUIL_STEPS="$(python - <<PY
equil_ps = float("${ATHERMAL_EQUIL_TIME_PS}")
timestep_ps = float("${TIMESTEP_PS}")
steps = round(equil_ps / timestep_ps)
if steps <= 0:
    raise SystemExit("Computed athermal equilibration steps must be positive")
print(int(steps))
PY
)"
fi

PY_LIVE_COMPUTE_ENERGY="False"
PY_LIVE_ENABLE_CUEQ="False"
PY_LIVE_ENABLE_OEQ="False"
if is_true "$LIVE_COMPUTE_ENERGY"; then
  PY_LIVE_COMPUTE_ENERGY="True"
fi
if is_true "$LIVE_ENABLE_CUEQ"; then
  PY_LIVE_ENABLE_CUEQ="True"
fi
if is_true "$LIVE_ENABLE_OEQ"; then
  PY_LIVE_ENABLE_OEQ="True"
fi

if [[ "$START_STAGE_NORMALIZED" == "all" ]]; then
  if [[ "$STRUCTURE_SOURCE_NORMALIZED" == "preprocessed" ]]; then
    prepare_structure_from_xyz "$PREPROCESSED_XYZ" "$STRUCTURE_DIR"
  else
    python "$WORKDIR/mp_fetcher.py" fetch "$MPID" \
      --api-key "$MP_API_KEY" \
      --supercell "${SUPERCELL_ARR[@]}" \
      --out-dir "$RUN_DIR" \
      --name "$STRUCTURE_NAME" \
      --no-infile
  fi

  [[ -d "$STRUCTURE_DIR" ]] || die "Expected structure directory was not created: $STRUCTURE_DIR"
else
  [[ -d "$STRUCTURE_DIR" ]] || die "Existing structure directory not found for START_STAGE=$START_STAGE: $STRUCTURE_DIR"
  if [[ "$IS_ZERO_TEMP" == "1" ]]; then
    [[ -f "$STRUCTURE_DIR/equilibrated.restart" ]] || die "Expected equilibrated restart not found for START_STAGE=$START_STAGE: $STRUCTURE_DIR/equilibrated.restart"
  else
    [[ -f "$STRUCTURE_DIR/equilibrated.data" ]] || die "Expected equilibrated data not found for START_STAGE=$START_STAGE: $STRUCTURE_DIR/equilibrated.data"
  fi
fi

cd "$STRUCTURE_DIR"
stage_models_for_run "$STRUCTURE_DIR"
stage_runtime_helpers "$STRUCTURE_DIR"
log "Staged LAMMPS model: $RUN_LAMMPS_MODEL"
log "Staged postprocess model: $RUN_POSTPROCESS_MODEL"
log "Staged live logger: $RUN_LIVE_LOGGER"

EQ_INPUT="in.01_equilibrate"
DRIVE_INPUT="in.02_hysteresis_5GHz"
RAW_EXTXYZ="hysteresis.raw.extxyz"
ANNOTATED_EXTXYZ="hysteresis.annotated.extxyz"

if [[ "$IS_ZERO_TEMP" == "1" ]]; then
cat >"$EQ_INPUT" <<EOF
# --- 01: BaTiO3 fixed-cell relaxation + athermal equilibration at ${TEMPERATURE_K} K with NVE + viscous damping (${STRUCTURE_SOURCE_NORMALIZED} start, supercell ${SUPER_TAG}) ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_data       structure.data

mass 1 15.999
mass 2 47.867
mass 3 137.327

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Ti Ba

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

$(if is_true "$RELAX_POSITIONS_BEFORE_MD"; then cat <<RELAX
min_style       ${MIN_STYLE}
minimize        ${MIN_ETOL} ${MIN_FTOL} ${MIN_MAXITER} ${MIN_MAXEVAL}

RELAX
fi)

timestep        ${TIMESTEP_PS}
reset_timestep  0

velocity        all set 0.0 0.0 0.0
fix             nve_all all nve
fix             viscous_all all viscous ${ATHERMAL_VISCOUS_GAMMA}

thermo          ${EQ_THERMO_EVERY}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz
thermo_modify   flush yes

dump            traj all custom ${EQ_DUMP_EVERY} equilibrate.lammpstrj id type xu yu zu fx fy fz
dump_modify     traj sort id

run             ${ATHERMAL_EQUIL_STEPS}

unfix           viscous_all
unfix           nve_all
write_restart   equilibrated.restart
write_data      equilibrated.data
EOF
else
cat >"$EQ_INPUT" <<EOF
# --- 01: BaTiO3 fixed-cell relaxation + equilibration at ${TEMPERATURE_K} K (${STRUCTURE_SOURCE_NORMALIZED} start, supercell ${SUPER_TAG}) ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_data       structure.data

mass 1 15.999
mass 2 47.867
mass 3 137.327

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Ti Ba

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

$(if is_true "$RELAX_POSITIONS_BEFORE_MD"; then cat <<RELAX
min_style       ${MIN_STYLE}
minimize        ${MIN_ETOL} ${MIN_FTOL} ${MIN_MAXITER} ${MIN_MAXEVAL}

RELAX
fi)

timestep        ${TIMESTEP_PS}
reset_timestep  0

velocity        all create ${TEMPERATURE_K} ${VELOCITY_SEED} mom yes dist gaussian
fix             mom all momentum 100 linear 1 1 1
fix             nvt_all all nvt temp ${TEMPERATURE_K} ${TEMPERATURE_K} ${TDAMP_PS}

thermo          ${EQ_THERMO_EVERY}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz
thermo_modify   flush yes

dump            traj all custom ${EQ_DUMP_EVERY} equilibrate.lammpstrj id type xu yu zu fx fy fz
dump_modify     traj sort id

run             ${EQUIL_STEPS}

unfix           nvt_all
unfix           mom
write_restart   equilibrated.restart
write_data      equilibrated.data
EOF
fi

if [[ "$IS_ZERO_TEMP" == "1" ]]; then
cat >"$DRIVE_INPUT" <<EOF
# --- 02: BaTiO3 finite-field hysteresis at ${TEMPERATURE_K} K (athermal NVE + viscous damping) ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_restart    equilibrated.restart

mass 1 15.999
mass 2 47.867
mass 3 137.327

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Ti Ba

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

timestep        ${TIMESTEP_PS}
reset_timestep  0

variable        E0 equal ${FIELD_MAX_Z}
variable        period equal ${FIELD_PERIOD_STEPS}
variable        phase equal ${FIELD_PHASE_RAD}
variable        Ex equal 0.0
variable        Ey equal 0.0
variable        Ez equal v_E0*cos(2.0*PI*step/v_period + v_phase)

python set_mace_efield here """
import os
from lammps import lammps

def set_mace_efield(lammps_ptr):
    lmp = lammps(ptr=lammps_ptr)
    ex = float(lmp.extract_variable("Ex", None, 0))
    ey = float(lmp.extract_variable("Ey", None, 0))
    ez = float(lmp.extract_variable("Ez", None, 0))
    os.environ["MACE_EFIELD"] = f"{ex},{ey},{ez}"
"""
fix             mace_efield all python/invoke 1 end_of_step set_mace_efield

fix             nve_all all nve
fix             viscous_all all viscous ${ATHERMAL_VISCOUS_GAMMA}

thermo          ${THERMO_EVERY}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz v_Ex v_Ey v_Ez
thermo_modify   flush yes

variable        step_out equal step
variable        time_out equal time
variable        temp_out equal temp
variable        pe_out equal pe
variable        ke_out equal ke
variable        etotal_out equal etotal
variable        press_out equal press
variable        pxx_out equal pxx
variable        pyy_out equal pyy
variable        pzz_out equal pzz
variable        pxy_out equal pxy
variable        pxz_out equal pxz
variable        pyz_out equal pyz
variable        vol_out equal vol
variable        lx_out equal lx
variable        ly_out equal ly
variable        lz_out equal lz
variable        xy_out equal xy
variable        xz_out equal xz
variable        yz_out equal yz

fix             thermo_out all print ${DUMP_EVERY} "\${step_out} \${time_out} \${temp_out} \${pe_out} \${ke_out} \${etotal_out} \${press_out} \${pxx_out} \${pyy_out} \${pzz_out} \${pxy_out} \${pxz_out} \${pyz_out} \${vol_out} \${lx_out} \${ly_out} \${lz_out} \${xy_out} \${xz_out} \${yz_out} \${Ex} \${Ey} \${Ez}" file hysteresis_thermo.tsv screen no title "# step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz Ex Ey Ez"

dump            traj all custom ${DUMP_EVERY} hysteresis.lammpstrj id type xu yu zu fx fy fz
dump_modify     traj sort id
EOF
else
cat >"$DRIVE_INPUT" <<EOF
# --- 02: BaTiO3 finite-field hysteresis at ${TEMPERATURE_K} K ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_data       equilibrated.data

mass 1 15.999
mass 2 47.867
mass 3 137.327

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Ti Ba

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

timestep        ${TIMESTEP_PS}
reset_timestep  0

variable        E0 equal ${FIELD_MAX_Z}
variable        period equal ${FIELD_PERIOD_STEPS}
variable        phase equal ${FIELD_PHASE_RAD}
variable        Ex equal 0.0
variable        Ey equal 0.0
variable        Ez equal v_E0*cos(2.0*PI*step/v_period + v_phase)

python set_mace_efield here """
import os
from lammps import lammps

def set_mace_efield(lammps_ptr):
    lmp = lammps(ptr=lammps_ptr)
    ex = float(lmp.extract_variable("Ex", None, 0))
    ey = float(lmp.extract_variable("Ey", None, 0))
    ez = float(lmp.extract_variable("Ez", None, 0))
    os.environ["MACE_EFIELD"] = f"{ex},{ey},{ez}"
"""
fix             mace_efield all python/invoke 1 end_of_step set_mace_efield

velocity        all create ${TEMPERATURE_K} ${VELOCITY_SEED} mom yes dist gaussian
fix             mom all momentum 100 linear 1 1 1
fix             nvt_all all nvt temp ${TEMPERATURE_K} ${TEMPERATURE_K} ${TDAMP_PS}

thermo          ${THERMO_EVERY}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz v_Ex v_Ey v_Ez
thermo_modify   flush yes

variable        step_out equal step
variable        time_out equal time
variable        temp_out equal temp
variable        pe_out equal pe
variable        ke_out equal ke
variable        etotal_out equal etotal
variable        press_out equal press
variable        pxx_out equal pxx
variable        pyy_out equal pyy
variable        pzz_out equal pzz
variable        pxy_out equal pxy
variable        pxz_out equal pxz
variable        pyz_out equal pyz
variable        vol_out equal vol
variable        lx_out equal lx
variable        ly_out equal ly
variable        lz_out equal lz
variable        xy_out equal xy
variable        xz_out equal xz
variable        yz_out equal yz

fix             thermo_out all print ${DUMP_EVERY} "\${step_out} \${time_out} \${temp_out} \${pe_out} \${ke_out} \${etotal_out} \${press_out} \${pxx_out} \${pyy_out} \${pzz_out} \${pxy_out} \${pxz_out} \${pyz_out} \${vol_out} \${lx_out} \${ly_out} \${lz_out} \${xy_out} \${xz_out} \${yz_out} \${Ex} \${Ey} \${Ez}" file hysteresis_thermo.tsv screen no title "# step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz Ex Ey Ez"

dump            traj all custom ${DUMP_EVERY} hysteresis.lammpstrj id type xu yu zu fx fy fz
dump_modify     traj sort id
EOF
fi

if is_true "$LIVE_RESPONSE_LOGGING"; then
cat >>"$DRIVE_INPUT" <<EOF

python log_macefield_response here """
import sys

sys.path.insert(0, "${STRUCTURE_DIR}")

from live_macefield_logger import LiveMACEFieldLogger, is_lammps_root

_live_logger = None

def log_macefield_response(lammps_ptr):
    global _live_logger

    if not is_lammps_root(lammps_ptr):
        return

    if _live_logger is None:
        _live_logger = LiveMACEFieldLogger(
            model_path="${RUN_POSTPROCESS_MODEL}",
            specorder=["O", "Ti", "Ba"],
            output_path="${LIVE_RESPONSE_EXTXYZ}",
            scalar_output_path="${LIVE_RESPONSE_TSV}",
            device="${LIVE_LOGGER_DEVICE}",
            dtype="${LIVE_LOGGER_DTYPE}",
            polarization_head="${MACEFIELD_HEAD}",
            response_head="${MACEFIELD_HEAD}",
            energy_head="${MACEFIELD_HEAD}",
            polarization_device="${LIVE_POLARIZATION_DEVICE}",
            response_device="${LIVE_RESPONSE_DEVICE}",
            energy_device="${LIVE_ENERGY_DEVICE}",
            enable_cueq=${PY_LIVE_ENABLE_CUEQ},
            enable_oeq=${PY_LIVE_ENABLE_OEQ},
            compute_energy=${PY_LIVE_COMPUTE_ENERGY},
        )

    _live_logger.log_step(lammps_ptr)
"""
fix             live_response all python/invoke ${LIVE_RESPONSE_EVERY} end_of_step log_macefield_response
EOF
fi

cat >>"$DRIVE_INPUT" <<EOF

run             ${DRIVE_STEPS}

EOF

if is_true "$LIVE_RESPONSE_LOGGING"; then
cat >>"$DRIVE_INPUT" <<EOF
unfix           live_response
EOF
fi

cat >>"$DRIVE_INPUT" <<EOF

unfix           thermo_out
EOF

if [[ "$IS_ZERO_TEMP" == "1" ]]; then
cat >>"$DRIVE_INPUT" <<EOF
unfix           viscous_all
unfix           nve_all
EOF
else
cat >>"$DRIVE_INPUT" <<EOF
unfix           nvt_all
unfix           mom
EOF
fi

cat >>"$DRIVE_INPUT" <<EOF
unfix           mace_efield
write_restart   hysteresis.restart
write_data      hysteresis.data
EOF

if [[ "$START_STAGE_NORMALIZED" == "all" ]]; then
  run_lammps_input "$EQ_INPUT" "equilibration" "0.0" "0.0" "$DRIVE_START_EZ"
fi
run_lammps_input "$DRIVE_INPUT" "finite-field hysteresis" "0.0" "0.0" "$DRIVE_START_EZ"

if [[ -f hysteresis.lammpstrj && -f hysteresis_thermo.tsv ]]; then
  log "Converting hysteresis.lammpstrj -> ${RAW_EXTXYZ}"
  python "$WORKDIR/lammps_dump_to_extxyz.py" \
    --dump hysteresis.lammpstrj \
    --thermo hysteresis_thermo.tsv \
    --output "$RAW_EXTXYZ" \
    --specorder O Ti Ba \
    --overwrite
fi

if is_true "$ANNOTATE_TRAJECTORY"; then
  if [[ -f "$RAW_EXTXYZ" ]]; then
    ANNOTATION_DEVICE_RESOLVED="$(select_annotation_device)"
    log "Annotation device: ${ANNOTATION_DEVICE_RESOLVED} (requested: ${ANNOTATION_DEVICE})"
    log "Backfilling response properties into ${ANNOTATED_EXTXYZ}"
    python "$WORKDIR/postprocess_macefield_xyz.py" \
      "$RAW_EXTXYZ" \
      --output "$ANNOTATED_EXTXYZ" \
      --model-path "$RUN_POSTPROCESS_MODEL" \
      --device "$ANNOTATION_DEVICE_RESOLVED" \
      --dtype "$ANNOTATION_DTYPE" \
      --head "$MACEFIELD_HEAD" \
      $(is_true "$ENABLE_CUEQ" && printf '%s' '--enable-cueq') \
      $(is_true "$ENABLE_OEQ" && printf '%s' '--enable-oeq') \
      $(is_true "$ANNOTATE_COMPUTE_ENERGY" && printf '%s' '--compute-energy')
  else
    log "Skipping annotation because ${RAW_EXTXYZ} was not created."
  fi
fi

log "Completed BaTiO3 hysteresis workflow."
if is_true "$LIVE_RESPONSE_LOGGING"; then
  log "Live response trajectory: $STRUCTURE_DIR/$LIVE_RESPONSE_EXTXYZ"
  log "Live response scalars:    $STRUCTURE_DIR/$LIVE_RESPONSE_TSV"
fi
if [[ -f "$RAW_EXTXYZ" ]]; then
  log "Raw extxyz trajectory:    $STRUCTURE_DIR/$RAW_EXTXYZ"
fi
if [[ -f "$ANNOTATED_EXTXYZ" ]]; then
  log "Annotated trajectory:     $STRUCTURE_DIR/$ANNOTATED_EXTXYZ"
fi
