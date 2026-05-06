# MP-Dielectric workflow

This folder contains the Materials Project dielectric dataset assembly, filtering, splitting, direct dielectric-model training, and Matbench overlap / refractive-index analysis.

Heavy artefacts for this workflow are distributed in the release assets `MACE-Field-MP-Dielectric.zip` and `MACE-Field-MP-Dielectrics.zip`; see the [latest release](https://github.com/mdi-group/2025-04-mace-field/releases/tag/resubmitted). Extract them into `scripts/Dielectrics/` for the original processed dataset bundle, training logs, and direct dielectric checkpoint.

Important data files:

- `MP-Dielectrics.extxyz`: raw assembled dataset.
- `MP-Dielectrics-filtered.extxyz`: filtered dataset used in the final paper workflow.
- `MP-Dielectrics-filtered-{train,valid,test}.xyz`: committed splits used by the training and foundation workflows.

## Key plots

### Matbench overlap and refractive-index check

<p align="center">
  <img src="MP-Dielectrics-filtered-matbench-refractive-index-parity.png" width="72%" alt="Matbench refractive-index parity">
</p>

The heavier BEC / polarizability parity plots used by the manuscript are generated through the foundation workflow in [../Foundation/README.md](../Foundation/README.md).

## Main entry points

Run all commands from this directory:

```bash
cd scripts/Dielectrics
```

### 1. Regenerate the MP-Dielectric dataset from the Materials Project API

```bash
python get-mp-dielectrics.py \
  --api-key "$MP_API_KEY" \
  --out MP-Dielectrics.extxyz \
  --write-filtered \
  --filtered-out MP-Dielectrics-filtered.extxyz \
  --write-splits \
  --split-prefix MP-Dielectrics-filtered \
  --verbose
```

This script can:

- fetch the raw dielectric dataset,
- filter high-force / pathological-BEC / pathological-dielectric entries,
- optionally deduplicate,
- write train/valid/test splits in `extxyz` format.

### 2. Train the direct dielectric model

```bash
bash train-dielectric.sh
```

Outputs:

- `results/MACE-Field-MP-Dielectrics_run-23_train.txt`
- `logs/MACE-Field-MP-Dielectrics_run-23.log`
- `checkpoints/`

### 3. Check overlap with Matbench dielectric and run refractive-index prediction

```bash
python check-matbench-dielectric-overlap.py \
  --mp-dataset MP-Dielectrics-filtered.extxyz \
  --predict-refractive-index \
  --model-path ../Foundation/MACEField-omat-dielectric.model \
  --head pt_head \
  --device cuda
```

Outputs:

- `MP-Dielectrics-filtered-vs-matbench-dielectric-overlap.csv`
- `MP-Dielectrics-filtered-vs-matbench-dielectric-overlap.json`
- `MP-Dielectrics-filtered-matbench-refractive-index-predictions.csv`
- `MP-Dielectrics-filtered-matbench-refractive-index-summary.json`
- `MP-Dielectrics-filtered-matbench-refractive-index-parity.png`

## Notes

- The final manuscript uses the filtered dataset and filtered splits committed here.
- In the lightweight repository, the full processed dataset bundle and direct dielectric training artefacts are expected to be restored from the release assets above.
- The foundation-model paper figures use these splits through the scripts in `../Foundation/`.
- The raw `matbench_dielectric.json.gz` cache is kept here so the overlap analysis can be rerun offline once downloaded.
