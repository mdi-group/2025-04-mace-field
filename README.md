# MACE-Field: General learning of the electric response of inorganic materials

This repository hosts the source files associated with our paper *“General learning of the Electric Response of Inorganic Materials”* (August 2025).  

MACE-Field is a field-aware, $O(3)$-equivariant interatomic potential built as a plug-in extension to MACE, enabling dielectric/ferroelectric/finite-field simulations of inorganic solids in a symmetry-consistent, ML-based framework.

---

## 🚀 What is MACE-Field

- **Field-aware interatomic potential**: MACE-Field injects a uniform external electric field into each message-passing layer via a Clebsch–Gordan tensor product coupling to latent spherical-tensor features, then perturbs them through equivariant residual mixing. This preserves the standard MACE readout while enabling consistent treatment of the electric field across chemistry.
- **Unified learning of electric response**: By learning a single electric enthalpy function $F(\{R\}, E)$ and differentiating it, MACE-Field delivers gauge-consistent predictions of key dielectric properties, e.g. polarisation $\mathbf{P}$, Born effective charges $Z^*$, and polarisability $\alpha$, along with the ability to carry out finite-field molecular dynamics or geometry relaxations.
- **Cross-chemistry transferability**: Our models cover a broad range of inorganic chemistries, spanning many different elements and structure types. As demonstrated in the paper, the approach generalises beyond single-material case studies.

---

## 📄 About the Paper

**Title:** *General learning of the Electric Response of Inorganic Materials* (2025)

**Abstract (short):**  
MACE-Field is a field-aware $O(3)$-equivariant interatomic potential that provides a compact, derivative-consistent route to dielectric properties (polarisation, Born effective charges, polarisability) and finite-field simulations of inorganic crystal solids. By injecting a uniform external field into message-passing layers via an equivariant tensor coupling, and inheriting pretrained MACE foundation weights, MACE-Field converts existing MACE force-field models into field-aware ones with minimal changes. We demonstrate its effectiveness by training cross-chemistry models for ferroelectric polarisation, BECs/polarisability for dielectric materials (spanning 81 elements), and performing finite-field MD / dielectric-constant simulations for prototypical materials (e.g. BaTiO3, α-quartz), achieving DFPT-grade accuracy. 

**Key results:**  
- Recovery of polarisation branches and spontaneous polarisation across non-polar → polar transformations.  
- Accurate predictions of Born effective charges $Z^*$ and electronic polarisability $\alpha$ over a diverse dataset of inorganic materials.  
- Successful finite-field molecular-dynamics and dielectric simulations for benchmark solids, including reproduction of ferroelectric hysteresis (BaTiO₃) and IR / Raman + dielectric spectra (α-quartz).

---

## 🧪 How to Use MACE-Field (via `mace-field` repo)

See the separate `mace-field` code repository for full instructions. In brief:  

1. Clone the `mace-field` repo.  
2. (Optional) Initialize from an existing pretrained MACE foundation model.  
3. Train or fine-tune on your dataset of interest (dielectrics, ferroelectrics, etc.).  
4. Use the trained model to compute dielectric properties, perform finite-field geometry relaxations or molecular dynamics, and extract polarisation, BECs, polarisability, etc.  

---

## 🎯 Why MACE-Field Matters

MACE-Field addresses a major gap in the ML-interatomic potential literature: the ability to correctly treat external electric fields, essential for modelling dielectric, ferroelectric, and nonlinear-optical materials. By combining the strengths of MACE (speed, accuracy, symmetry-aware message passing) with a minimal and physically consistent extension for field coupling, MACE-Field offers a scalable, data-efficient, and widely transferable framework for routine finite-field simulations of inorganic materials. With growing interest in high-throughput materials discovery, this capability opens the door to exploring new dielectrics, ferroelectrics, and multifunctional materials at scale.  

---

## 📚 Citation

If you use MACE-Field in your work, please cite:

```bibtex
@misc{martin2025generallearningelectricresponse,
  title={General Learning of the Electric Response of Inorganic Materials},
  author={Martin, Bradley A. A. and Ganose, Alex M. and Kapil, Venkat and Li, Tingwei and Butler, Keith T.},
  year={2025},
  eprint={2508.17870},
  archivePrefix={arXiv},
}
```
---

## 🔗 Links

- [mace-field repository (code, examples, pretrained models)](https://github.com/mdi-group/mace-field) 

---

