#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DIELECTRIC_DATA="${SCRIPT_DIR}/../Dielectrics/MP-Dielectrics-filtered.extxyz"
FERROELECTRIC_DATA="${SCRIPT_DIR}/../Ferroelectrics/MP-Ferroelectrics.xyz"
FIELD_DATA="${SCRIPT_DIR}/field-data.xyz"

DIELECTRIC_DATA="${DIELECTRIC_DATA}" \
FERROELECTRIC_DATA="${FERROELECTRIC_DATA}" \
FIELD_DATA="${FIELD_DATA}" \
python - <<'PY'
import os
from pathlib import Path

from ase.io import read, write

sources = [
    Path(os.environ["DIELECTRIC_DATA"]),
    Path(os.environ["FERROELECTRIC_DATA"]),
]
output = Path(os.environ["FIELD_DATA"])

frames = []
for source in sources:
    if not source.exists():
        raise FileNotFoundError(f"Missing replay-source dataset: {source}")
    atoms = read(source, ":")
    atoms_list = atoms if isinstance(atoms, list) else [atoms]
    for frame in atoms_list:
        copied = frame.copy()
        copied.info["replay_source_dataset"] = source.name
        frames.append(copied)

output.parent.mkdir(parents=True, exist_ok=True)
write(output, frames, format="extxyz")
print(f"Wrote {len(frames)} finetuning reference frames to {output}")
PY

#python -m mace.cli.fine_tuning_select \
#    --configs_pt "mp_traj_combined.xyz" \
#    --configs_ft "field-data.xyz" \
#    --num_samples 10000 \
#    --model "mace-mp-0b3-medium.model" \
#    --output "mp_traj_selected.xyz" \
#    --filtering_type combinations \
#    --head_pt default \
#    --head_ft mp-dielectric-ferroelectric \
#    --device cuda \
#    --descriptors "descriptors.npy" \


torchrun --standalone --nproc_per_node="gpu" \
    "../../../mace/mace-field/mace/cli/run_train.py" \
    --config "config.yaml"
    
