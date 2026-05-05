## Allegro-pol quartz reference data

This folder collects the Allegro-pol reference files I could locate directly from the
public repository and paper for the `alpha`-SiO2 / quartz comparisons.

Primary public sources:

- Nature Communications paper:
  `https://www.nature.com/articles/s41467-025-59304-1`
- Allegro-pol repository:
  `https://github.com/mir-group/allegro-pol`

### Exact repo-backed files copied here

From `scripts/5.Vibrational/SiO2/DFPT/` in the Allegro-pol repository:

- `DFPT/SiO2-IR-dfpt.dat`
- `DFPT/SiO2-epsre-dfpt.dat`
- `DFPT/SiO2-epsim-dfpt.dat`

These are the cleanest machine-readable DFPT reference curves I found for:

- IR intensities
- Re epsilon(omega)
- Im epsilon(omega)

From `scripts/4.Dielectric/SiO2/DFT/`:

- `DFT/SiO2-E0.out`
- `DFT/SiO2-Ez-1e-3.out`

These are the underlying DFT structural-relaxation outputs used by Allegro-pol's
dielectric-constant analysis script.

From the Allegro-pol ML example folders:

- `ML/SiO2-mlmd.dat`
- `ML/SiO2-sc222_1e-3.dat`

These are included for completeness because the plotting scripts in the public repo
consume them directly.

### Exact repo-backed figures copied here

- `SiO2-mlmd.pdf`
- `SiO2-dielectric-relaxation.pdf`
- `SiO2-vibrational-overview.png`

### Compact table extracted from the paper/manuscript context

- `quartz_reference_table.csv`

This is not a raw Allegro-pol source-data export. It is a compact tabulation of:

- Allegro-pol Table 1 values for the three main IR peaks and dielectric constants
- representative quartz DFPT and experimental dielectric constants quoted alongside
  the comparison in the local manuscript

### What I did **not** find

I did not find a separate public machine-readable file in Allegro-pol for:

- full experimental quartz IR curves
- full experimental quartz Raman curves
- full experimental epsilon(omega) curves

So the exact public files recovered here are mainly the Allegro-pol DFPT and DFT
reference curves, plus the compact benchmark numbers saved in
`quartz_reference_table.csv`.
