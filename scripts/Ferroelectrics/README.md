# MP-Ferroelectric workflow

This folder contains the cross-chemistry ferroelectric dataset assembly, direct ferroelectric-model training, and branch-aware polarisation analysis used for the folded parity and spontaneous-polarisation results.

Heavy artefacts for this workflow are distributed in the release asset `MACE-Field-MP-Ferroelectrics.zip`; see [../../GITHUB_RELEASE.md](../../GITHUB_RELEASE.md). Extract that archive into `scripts/Ferroelectrics/` for the original dataset splits, checkpoint, and training logs.

Important data and model files:

- `MP-Ferroelectrics.xyz`: combined ferroelectric dataset.
- `MP-Ferroelectrics-{train,valid,test}.xyz`: committed splits used in the paper.
- `MACE-Field-MP-Ferroelectrics.model`: direct ferroelectric model.
- `MACEField-omat-dielectric.model`: OMAT-based foundation checkpoint used for comparison.

## Key plots

### Training curve

<p align="center">
  <img src="results/MACE-Field-MP-Ferroelectrics_run-23_train_Default_stage_one.png" width="72%" alt="MP-Ferroelectrics training curve">
</p>

### Folded parity and spontaneous polarisation

<p align="center">
  <img src="analysis_outputs/polarization_branches/MACE-Field-MP-Ferroelectrics_parity_folded_splits.png" width="49%" alt="Direct folded parity">
  <img src="analysis_outputs/polarization_branches/MACEField-omat-dielectric_parity_folded_splits.png" width="49%" alt="Foundation folded parity">
</p>

<p align="center">
  <img src="analysis_outputs/polarization_branches/MACE-Field-MP-Ferroelectrics_spontaneous_polarization_folded_splits.png" width="49%" alt="Direct spontaneous polarization parity">
  <img src="analysis_outputs/polarization_branches/MACEField-omat-dielectric_spontaneous_polarization_folded_splits.png" width="49%" alt="Foundation spontaneous polarization parity">
</p>

### Branch pathways and distributions

<p align="center">
  <img src="analysis_outputs/polarization_branches/MACE-Field-MP-Ferroelectrics_path_branches_000.png" width="49%" alt="Direct branch pathway example">
  <img src="analysis_outputs/polarization_branches/MACEField-omat-dielectric_path_branches_000.png" width="49%" alt="Foundation branch pathway example">
</p>

<p align="center">
  <img src="analysis_outputs/polarization_branches/MACE-Field-MP-Ferroelectrics_fractional_branch_distribution.png" width="49%" alt="Direct fractional branch distribution">
  <img src="analysis_outputs/polarization_branches/MACEField-omat-dielectric_fractional_branch_distribution.png" width="49%" alt="Foundation fractional branch distribution">
</p>

## Main entry points

Run all commands from this directory:

```bash
cd scripts/Ferroelectrics
```

### 1. Regenerate the MP-Ferroelectric dataset from MPContribs / Materials Project

```bash
python get_ferroelectric_dataset.py \
  --api-key "$MP_API_KEY" \
  --out MP-Ferroelectrics.xyz \
  --write-splits \
  --split-prefix MP-Ferroelectrics \
  --no-allow-branch-cross-split \
  --verbose
```

The committed splits were created with branch-aware grouping so that branch/path leakage is avoided.

### 2. Train the direct ferroelectric model

```bash
bash train_ferroelectrics.sh
```

Outputs:

- `results/MACE-Field-MP-Ferroelectrics_run-23_train.txt`
- `logs/MACE-Field-MP-Ferroelectrics_run-23.log`
- `checkpoints/`

### 3. Run the folded-branch analysis

With the defaults, the analysis script evaluates both the direct ferroelectric model and the OMAT-based foundation model committed in this folder.

```bash
python polarization_branches.py
```

Key outputs:

- `analysis_outputs/polarization_branches/*_parity_folded_splits.png`
- `analysis_outputs/polarization_branches/*_spontaneous_polarization_folded_splits.png`
- `analysis_outputs/polarization_branches/*_fractional_branch_distribution.png`
- `analysis_outputs/polarization_branches/combined_polarization_summary.csv`

## Notes

- The folding / branch matching here is the paper workflow used for the ferroelectric parity figures.
- In a lightweight checkout, the large dataset and training artefacts are expected to come from the release asset rather than the main repository history.
- The final manuscript copies of the main folded parity plots live in `../../figures/`.
- The foundation comparison in this folder uses the OMAT-based multihead checkpoint rather than the older MP-traj-era model.
