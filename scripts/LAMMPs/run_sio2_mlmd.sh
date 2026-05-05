#!/usr/bin/env bash

# Workflow launcher for bulk SiO2 (mp-7000) MLMD with MACEField.
#
# What it does:
#   1. starts from a curated SiO2 structure (or optionally fetches from MP)
#   2. builds a configurable supercell while preserving the input lattice
#   3. relaxes atomic positions at fixed cell, then equilibrates at 300 K with MACEField in LAMMPS
#   4. runs a 200 ps NVT production trajectory at 300 K
#   5. logs response properties live during production via fix python/invoke
#   6. optionally converts the LAMMPS dump + thermo sidecar into extxyz
#   7. optionally backfills response properties after the run as a fallback
#
# Common overrides:
#   SUPERCELL="1 1 1" LAMMPS_ARGS="-np 4" ./run_sio2_mlmd.sh
#   STRUCTURE_SOURCE=mp MP_API_KEY=... ./run_sio2_mlmd.sh
#   MACEFIELD_MODEL=models/MACEField-omat-dielectric.model ./run_sio2_mlmd.sh
#   MODEL_VARIANT=finetuned ./run_sio2_mlmd.sh
#   POLARIZATION_DEVICE=cuda:0 RESPONSE_DEVICE=cuda:1 ./run_sio2_mlmd.sh
#   START_STAGE=02 RUN_TAG=SiO2-mp-7000-sc1x1x1-300K-200ps-YYYY-MM-DD_HHMMSS ./run_sio2_mlmd.sh

set -eo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SIF:-$WORKDIR/macefield-lammps.sif}"
CONDA_SH="/home/brad/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-MACEField}"
LOGROOT="${LOGROOT:-$WORKDIR/logs}"

MPID="${MPID:-mp-7000}"
STRUCTURE_NAME="${STRUCTURE_NAME:-SiO2-mp-7000}"
STRUCTURE_SOURCE="${STRUCTURE_SOURCE:-preprocessed}"
PREPROCESSED_XYZ="${PREPROCESSED_XYZ:-/home/brad/repositories/2025-04-mace-field/scripts/SiO2/SiO2-preprocessed.xyz}"
SUPERCELL="${SUPERCELL:-1 1 1}"
RUN_PARENT="${RUN_PARENT:-$WORKDIR/MD/runs}"
RUN_TAG="${RUN_TAG:-}"
START_STAGE="${START_STAGE:-all}"
MODEL_VARIANT="${MODEL_VARIANT:-foundation}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
BACKGROUND_CHILD="${BACKGROUND_CHILD:-0}"
BACKGROUND_LOG_ONLY="${BACKGROUND_LOG_ONLY:-0}"

TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
TIMESTEP_PS="${TIMESTEP_PS:-0.002}"
EQUIL_TIME_PS="${EQUIL_TIME_PS:-10.0}"
PROD_TIME_PS="${PROD_TIME_PS:-200.0}"
EQUIL_STEPS="${EQUIL_STEPS:-}"
PROD_STEPS="${PROD_STEPS:-}"
TDAMP_PS="${TDAMP_PS:-0.1}"
PDAMP_PS="${PDAMP_PS:-1.0}"
NEIGHBOR_SKIN="${NEIGHBOR_SKIN:-2.0}"
VELOCITY_SEED="${VELOCITY_SEED:-4928459}"
RELAX_POSITIONS_BEFORE_MD="${RELAX_POSITIONS_BEFORE_MD:-1}"
MIN_STYLE="${MIN_STYLE:-cg/kk}"
MIN_ETOL="${MIN_ETOL:-1.0e-10}"
MIN_FTOL="${MIN_FTOL:-1.0e-10}"
MIN_MAXITER="${MIN_MAXITER:-20000}"
MIN_MAXEVAL="${MIN_MAXEVAL:-200000}"

EQ_DUMP_EVERY="${EQ_DUMP_EVERY:-100}"
EQ_THERMO_EVERY="${EQ_THERMO_EVERY:-100}"
DUMP_EVERY="${DUMP_EVERY:-1}"
THERMO_EVERY="${THERMO_EVERY:-$DUMP_EVERY}"

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
LIVE_RESPONSE_EXTXYZ="${LIVE_RESPONSE_EXTXYZ:-production.live.extxyz}"
LIVE_RESPONSE_TSV="${LIVE_RESPONSE_TSV:-production.live.tsv}"
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
DEFAULT_FINETUNED_POSTPROCESS_MODEL="/home/brad/repositories/2025-04-mace-field/scripts/SiO2/MACE-field-SiO2.model"
DEFAULT_FINETUNED_LAMMPS_MODEL="/home/brad/repositories/2025-04-mace-field/scripts/SiO2/MACE-field-SiO2.model-mliap_lammps.pt"

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
  local best_gpu="" best_count="" gpu count

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
    02|2|prod|production|resume) printf '02' ;;
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
  local -a apptainer_cmd env_args lmp_args

  read -r -a lmp_args <<< "$LAMMPS_ARGS"

  if is_true "$USE_SUDO"; then
    apptainer_cmd=(sudo "$APPTAINER_BIN")
  else
    apptainer_cmd=("$APPTAINER_BIN")
  fi

  env_args=(
    --env "MACE_EFIELD_MODE=env"
    --env "MACE_EFIELD=${EX_FIELD},${EY_FIELD},${EZ_FIELD}"
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

START_STAGE_NORMALIZED="$(normalize_stage "$START_STAGE")" || die "START_STAGE must be one of: all, 01, 02, production, resume"
MODEL_VARIANT_NORMALIZED="$(normalize_model_variant "$MODEL_VARIANT")" || die "MODEL_VARIANT must be one of: foundation, finetuned"
STRUCTURE_SOURCE_NORMALIZED="$(normalize_structure_source "$STRUCTURE_SOURCE")" || die "STRUCTURE_SOURCE must be one of: preprocessed, mp"
if [[ -z "$MACEFIELD_HEAD" ]]; then
  MACEFIELD_HEAD="$(default_head_for_variant "$MODEL_VARIANT_NORMALIZED")" || die "Could not determine default MACEFIELD_HEAD"
fi

if [[ -z "$EQUIL_STEPS" ]]; then
  EQUIL_STEPS="$(python - <<PY
equil_ps = float("${EQUIL_TIME_PS}")
timestep_ps = float("${TIMESTEP_PS}")
steps = round(equil_ps / timestep_ps)
if steps <= 0:
    raise SystemExit("Computed equilibration steps must be positive")
print(int(steps))
PY
)"
fi

if [[ -z "$PROD_STEPS" ]]; then
  PROD_STEPS="$(python - <<PY
prod_ps = float("${PROD_TIME_PS}")
timestep_ps = float("${TIMESTEP_PS}")
steps = round(prod_ps / timestep_ps)
if steps <= 0:
    raise SystemExit("Computed production steps must be positive")
print(int(steps))
PY
)"
fi

SUPER_TAG="${SUPERCELL_ARR[0]}x${SUPERCELL_ARR[1]}x${SUPERCELL_ARR[2]}"
if [[ -z "$RUN_TAG" ]]; then
  [[ "$START_STAGE_NORMALIZED" == "all" ]] || die "RUN_TAG is required when START_STAGE=$START_STAGE"
  RUN_TAG="${STRUCTURE_NAME}-sc${SUPER_TAG}-300K-200ps-$(date '+%F_%H%M%S')"
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
log "Timestep: ${TIMESTEP_PS} ps"
log "Equilibration: ${EQUIL_TIME_PS} ps (${EQUIL_STEPS} steps)"
log "Production: ${PROD_TIME_PS} ps (${PROD_STEPS} steps)"
if [[ "${SUPERCELL_ARR[*]}" != "14 14 14" ]]; then
  log "Using a reduced SiO2 supercell for practicality; Allegro-pol used 14 14 14 (24696 atoms)."
fi
log "Production sampling: every ${DUMP_EVERY} step(s)"
if is_true "$LIVE_RESPONSE_LOGGING"; then
  log "Live response logging: every ${LIVE_RESPONSE_EVERY} step(s) -> ${LIVE_RESPONSE_EXTXYZ}"
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
  [[ -f "$STRUCTURE_DIR/equilibrated.restart" ]] || die "Expected restart not found for START_STAGE=$START_STAGE: $STRUCTURE_DIR/equilibrated.restart"
fi

cd "$STRUCTURE_DIR"
stage_models_for_run "$STRUCTURE_DIR"
stage_runtime_helpers "$STRUCTURE_DIR"
log "Staged LAMMPS model: $RUN_LAMMPS_MODEL"
log "Staged postprocess model: $RUN_POSTPROCESS_MODEL"
log "Staged live logger: $RUN_LIVE_LOGGER"

EQ_INPUT="in.01_equilibrate_300K"
PROD_INPUT="in.02_mlmd_300K"
RAW_EXTXYZ="production.raw.extxyz"
ANNOTATED_EXTXYZ="production.annotated.extxyz"

cat >"$EQ_INPUT" <<EOF
# --- 01: bulk SiO2 fixed-cell relaxation + NVT equilibration at ${TEMPERATURE_K} K (${STRUCTURE_SOURCE_NORMALIZED} start, supercell ${SUPER_TAG}) ---

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

cat >"$PROD_INPUT" <<EOF
# --- 02: 200 ps bulk SiO2 MLMD at ${TEMPERATURE_K} K with fixed-cell NVT dynamics and per-step response logging hooks ---

units           metal
atom_style      atomic
boundary        p p p
newton          on
box             tilt large

read_restart    equilibrated.restart

mass 1 15.999
mass 2 28.085

pair_style      mliap unified ${RUN_LAMMPS_MODEL} 0
pair_coeff      * * O Si

neighbor        ${NEIGHBOR_SKIN} bin
neigh_modify    every 1 delay 0 check yes

variable        Ex equal ${EX_FIELD}
variable        Ey equal ${EY_FIELD}
variable        Ez equal ${EZ_FIELD}

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

timestep        ${TIMESTEP_PS}
reset_timestep  0

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

fix             thermo_out all print ${DUMP_EVERY} "\${step_out} \${time_out} \${temp_out} \${pe_out} \${ke_out} \${etotal_out} \${press_out} \${pxx_out} \${pyy_out} \${pzz_out} \${pxy_out} \${pxz_out} \${pyz_out} \${vol_out} \${lx_out} \${ly_out} \${lz_out} \${xy_out} \${xz_out} \${yz_out} \${Ex} \${Ey} \${Ez}" file production_thermo.tsv screen no title "# step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol lx ly lz xy xz yz Ex Ey Ez"

dump            traj all custom ${DUMP_EVERY} production.lammpstrj id type xu yu zu fx fy fz
dump_modify     traj sort id
EOF

if is_true "$LIVE_RESPONSE_LOGGING"; then
cat >>"$PROD_INPUT" <<EOF

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
            specorder=["O", "Si"],
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

cat >>"$PROD_INPUT" <<EOF

run             ${PROD_STEPS}

EOF

if is_true "$LIVE_RESPONSE_LOGGING"; then
cat >>"$PROD_INPUT" <<EOF
unfix           live_response
EOF
fi

cat >>"$PROD_INPUT" <<EOF

unfix           thermo_out
unfix           nvt_all
unfix           mom
unfix           mace_efield
write_restart   production.restart
write_data      production.data
EOF

if [[ "$START_STAGE_NORMALIZED" == "all" ]]; then
  run_lammps_input "$EQ_INPUT" "equilibration"
fi
run_lammps_input "$PROD_INPUT" "production"

python "$WORKDIR/lammps_dump_to_extxyz.py" \
  --dump production.lammpstrj \
  --thermo production_thermo.tsv \
  --output "$RAW_EXTXYZ" \
  --specorder O Si \
  --overwrite

if is_true "$ANNOTATE_TRAJECTORY"; then
  ANNOTATION_DEVICE_RESOLVED="$(select_annotation_device)"
  log "Annotation device: ${ANNOTATION_DEVICE_RESOLVED} (requested: ${ANNOTATION_DEVICE})"
  post_cmd=(
    python "$WORKDIR/postprocess_macefield_xyz.py"
    "$RAW_EXTXYZ"
    --model-path "$RUN_POSTPROCESS_MODEL"
    --device "$ANNOTATION_DEVICE_RESOLVED"
    --dtype "$ANNOTATION_DTYPE"
    --electric-field "${EX_FIELD},${EY_FIELD},${EZ_FIELD}"
    --head "$MACEFIELD_HEAD"
    --output "$ANNOTATED_EXTXYZ"
    --overwrite
  )
  if is_true "$ANNOTATE_COMPUTE_ENERGY"; then
    post_cmd+=(--compute-energy)
  fi
  if is_true "$ENABLE_CUEQ"; then
    post_cmd+=(--enable-cueq)
  fi
  if is_true "$ENABLE_OEQ"; then
    post_cmd+=(--enable-oeq)
  fi

  "${post_cmd[@]}"
fi

log "Finished."
log "Structure directory: $STRUCTURE_DIR"
log "LAMMPS dump: $STRUCTURE_DIR/production.lammpstrj"
log "Raw extxyz: $STRUCTURE_DIR/$RAW_EXTXYZ"
if is_true "$LIVE_RESPONSE_LOGGING"; then
  log "Live response extxyz: $STRUCTURE_DIR/$LIVE_RESPONSE_EXTXYZ"
  log "Live response table: $STRUCTURE_DIR/$LIVE_RESPONSE_TSV"
fi
if is_true "$ANNOTATE_TRAJECTORY"; then
  log "Annotated extxyz: $STRUCTURE_DIR/$ANNOTATED_EXTXYZ"
fi
log "Log file: $LOGFILE"
