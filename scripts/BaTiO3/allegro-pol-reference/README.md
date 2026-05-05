## Allegro-pol BaTiO3 reference data

This folder collects the Allegro-pol BaTiO3 hysteresis references I could recover
 from the public repository, paper, and the existing handoff bundle.

Primary public sources:

- Nature Communications paper:
  `https://www.nature.com/articles/s41467-025-59304-1`
- Allegro-pol repository:
  `https://github.com/mir-group/allegro-pol`

### Exact repo-backed files copied here

From `scripts/6.Hysteresis/BaTiO3-8640-T300/` in the Allegro-pol repository:

- `BaTiO3-1.dat` through `BaTiO3-10.dat`
- `BaTiO3-8640-T300.pdf`
- `BaTiO3-hysteresis-overview.png`

These are the public Allegro-pol MLMD hysteresis reference files for:

- 8640-atom BaTiO3
- 300 K
- sinusoidal electric field

### Figure-digitized files copied here

- `allegro_pol_bto_fig3a_dft_digitized.csv`
- `allegro_pol_bto_fig3a_dft_digitized_dense.csv`

This corresponds to the 135-atom, 0 K DFT hysteresis loop discussed around Fig. 3a
in the paper. I did **not** find a raw public machine-readable source file for that
loop in the Allegro-pol repository. These CSVs therefore remain approximate
digitizations from the published figure / prior handoff, and should be treated as
figure-extracted rather than exact source data.

The `*_dense.csv` file is a later denser extraction from the actual Nature Fig. 3
image and is the version preferred automatically by `plot_hysteresis.py`. Because
some upper-branch triangles are partly occluded by the legend/annotations in the
published panel, the dense file uses the visually extracted lower branch plus the
figure's stated hysteresis symmetry to complete the upper branch to 17 points.

### What I did **not** find

I did not find a public raw data file for:

- the 135-atom, 0 K DFT finite-field hysteresis loop from Fig. 3a

So the exact public references here are the 300 K MLMD hysteresis files, while the
Fig. 3a DFT loop is included only as an explicitly labeled digitized approximation.
