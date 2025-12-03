import os
from mp_api.client import MPRester
from pymatgen.core import Molecule

# Fetch and prepare molecule class
class MaterialFetcher:
    """
    Fetches a material structure from the Materials Project API,
    builds a supercell, and writes an ASE‑readable XYZ file with PBC.
    """
    def __init__(self, api_key: str):
        self.mpr = MPRester(api_key=api_key)
    
    def fetch(self, mpid: str, supercell=(2,2,2), out_dir="MD") -> dict:
        # 1) Download and supercell
        struct = self.mpr.get_structure_by_material_id(mpid)
        struct.make_supercell(supercell)
        
        # 2) Prepare output directory
        formula = struct.composition.reduced_formula
        name = f"{formula}-{mpid}"
        target_dir = os.path.join(out_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 3) Build lattice comment (diagonal only)
        mat = struct.lattice.matrix
        lattice_flat = " ".join(map(str, mat.flatten()))
        lattice_comment = f'Lattice="{lattice_flat}" pbc="T T T"'
        
        # 4) Convert to pymatgen Molecule for XYZ export
        mol = Molecule(
            species=struct.species,
            coords=struct.cart_coords,
            charge=struct.charge,
            site_properties=struct.site_properties,
        )
        xyz_str = mol.to(fmt="xyz")
        lines = xyz_str.splitlines()
        # Insert lattice line after atom count
        xyz_full = "\n".join([lines[0], lattice_comment, *lines[2:]])
        
        out_path = os.path.join(target_dir, f"{name}.xyz")
        with open(out_path, "w") as f:
            f.write(xyz_full)
        
        return {"name": name, "dir": target_dir, "xyz": out_path}

