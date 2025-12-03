import os
import time

from mace.calculators import MACECalculator

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

import numpy as np
import torch

from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase import units
from ase.visualize import view

# MD runner 
class MACEFieldMD:
    """
    ASE MD runner using field‑aware MACE, with arbitrary time‑dependent field and
    notebook‑friendly live plotting of energy, temperature, and field components.
    """
    def __init__(self, model_path: str, device="cpu", dtype="float32", head="Default"):
        self.calc = MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype=dtype,
            head=head,
            model_type="ScaleShiftFieldMACE",
            
        )

    def run(
        self,
        xyz_path: str,
        temperature: float,
        timestep_fs: float,
        total_steps: int,
        log_interval: int,
        field_func: callable = None,
        visualize: bool = False,
    ):
        atoms = read(xyz_path)
        atoms.calc = self.calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms)
        ZeroRotation(atoms)

        dyn = Langevin(
            atoms,
            timestep_fs * units.fs,
            temperature_K=temperature,
            friction=0.1,
        )

        # Prepare logs
        logs = {k: [] for k in (
            "time_fs","energy","T","field","pol","alpha","stress","lattice"
        )}
        traj_file = xyz_path.replace(".xyz", "_trace.xyz")
        os.system(f"rm -f {traj_file}")

        # Titles for the 3×3 grid
        titles = [
            "Energy per Atom (eV)", "Temperature (K)", "Field Components (V/Å)",
            "polarisation Components", "polarisation vs E", "polarisability (e/Å/V)",
            "Stress Tensor", "Lattice Constants (Å)", "Lattice Angles (°)"
        ]
        if visualize:
            fig, axes = plt.subplots(3, 3, figsize=(14, 12), tight_layout=True)

        def _logger():
            # 1) Step & time
            step = len(logs["time_fs"])
            t_fs = dyn.get_time() / units.fs

            # print("step:", step, "time:", t_fs)

            # 2) Compute & assign field
            efield = field_func(step, t_fs) if field_func else [0.0,0.0,0.0]
            dyn.atoms.calc.electric_field = torch.tensor(efield)
            res = dyn.atoms.calc.results

            # 3) Write trajectory
            dyn.atoms.info["MACE_electric_field"] = efield
            dyn.atoms.info["MACE_polarisation"]   = res["polarisation"]
            dyn.atoms.arrays["MACE_becs"]         = res["becs"]
            dyn.atoms.info["MACE_polarisability"] = res["polarisability"]
            dyn.atoms.write(traj_file, append=True)

            # 4) Log data
            logs["time_fs"].append(t_fs)
            logs["energy"].append(dyn.atoms.get_potential_energy()/len(atoms))
            logs["T"].append(atoms.get_temperature())
            logs["field"].append(efield)
            logs["pol"].append(res["polarisation"])
            # Flatten full 3x3 α to 9
            alpha_full = np.array(res["polarisability"]).reshape(9)
            logs["alpha"].append(alpha_full)
            logs["stress"].append(atoms.get_stress())
            logs["lattice"].append(atoms.get_cell_lengths_and_angles())

            if visualize:
                # Clear & reset each subplot
                for ax, title in zip(axes.flatten(), titles):
                    ax.clear()
                    ax.set_title(title, fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.4)
                    ax.tick_params(labelsize=10)

                t = np.array(logs["time_fs"])
                fld = np.array(logs["field"])
                pol = np.array(logs["pol"])
                strss = np.array(logs["stress"])
                lat = np.array(logs["lattice"])
                alpha = np.array(logs["alpha"])

                # Panel (0,0): Energy
                axes[0,0].plot(t, logs["energy"], lw=1.5, marker='o', ms=3, color='tab:blue')
                axes[0,0].set_xlabel("fs"); axes[0,0].set_ylabel("eV")

                # Panel (0,1): Temperature
                axes[0,1].plot(t, logs["T"], lw=1.5, marker='s', ms=3, color='tab:red')
                axes[0,1].set_xlabel("fs"); axes[0,1].set_ylabel("K")

                # Panel (0,2): Field components
                for i,label,c in zip(range(3), ['Ex','Ey','Ez'], ['r','g','b']):
                    axes[0,2].plot(t, fld[:,i], lw=1.25, marker='o', ms=3,  label=label, color=c)
                axes[0,2].legend(fontsize=9)

                # Panel (1,0): polarisation components
                for i,label,c in zip(range(3), ['Px','Py','Pz'], ['r','g','b']):
                    axes[1,0].plot(t, pol[:,i], lw=1.25, marker='o', ms=3,  label=label, color=c)
                axes[1,0].legend(fontsize=9)

                # Panel (1,1): polarisation vs Ez
                for i,label,c in zip(range(3), ['Px','Py','Pz'], ['r','g','b']):
                    axes[1,1].scatter(fld[:,2], pol[:,i], s=10, label=label, alpha=0.7, color=c)
                axes[1,1].legend(fontsize=9)

                # Panel (1,2): All 9 polarisability components
                alpha_labels = ['αxx','αxy','αxz','αyx','αyy','αyz','αzx','αzy','αzz']
                colors = plt.cm.tab10.colors
                for i,label in enumerate(alpha_labels):
                    axes[1,2].plot(t, alpha[:,i], lw=1.2, marker='o', ms=3, label=label, color=colors[i % len(colors)])
                axes[1,2].legend(fontsize=7, ncol=3, loc='upper right')

                # Panel (2,0): Stress tensor
                stress_labels = ['σxx','σyy','σzz','σyz','σxz','σxy']
                for i,label,c in zip(range(6), stress_labels, ['r','g','b','c','m','y']):
                    axes[2,0].plot(t, strss[:,i], lw=1, marker='o', ms=3, label=label, color=c)
                axes[2,0].legend(fontsize=8, ncol=2)

                # Panel (2,1): Lattice constants
                for i,label,c in zip(range(3), ['a','b','c'], ['r','g','b']):
                    axes[2,1].plot(t, lat[:,i], lw=1.5, marker='o', ms=3, label=label, color=c)
                axes[2,1].legend(fontsize=9)

                # Panel (2,2): Lattice angles
                for idx,label,c in zip(range(3,6), ['α','β','γ'], ['r','g','b']):
                    axes[2,2].plot(t, lat[:,idx], lw=1.5, marker='o', ms=3, label=label, color=c)
                axes[2,2].legend(fontsize=9)

                # Clear and display both figure and structure widget
                clear_output(wait=True)
                display(fig)
                display(view(dyn.atoms, viewer='x3d'))

        dyn.attach(_logger, interval=log_interval)
        t0 = time.time()
        dyn.run(total_steps)
        print(f"MD done in {(time.time()-t0)/60:.2f} min")
        return logs, traj_file
    
    