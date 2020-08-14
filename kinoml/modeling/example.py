"""
Draft to show how modeling classes can be implemented in KinoML
"""

from ..core.proteins import ProteinStructure

class ProteinAtomCount:

    def __init__(self, atom_names=None):
        self.atom_names = atom_names

    def execute(self, component: ProteinStructure):
        u = component.universe
        if self.atom_names:
            return len([a for a in u.atoms if a.name in self.atom_names])
        return u.n_atoms

    __call__ = execute


if __name__ == "__main__":
    import sys

    structure = ProteinStructure.from_file(sys.argv[1])
    counter = ProteinAtomCount()
    n_atoms = counter.execute(structure)
    print(f"Structure {sys.argv[1]} has {n_atoms} atoms.")