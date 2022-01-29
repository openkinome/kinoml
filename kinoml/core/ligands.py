"""
``MolecularComponent`` objects that represent ligand-like entities.
"""

import logging

from .components import BaseLigand


logger = logging.getLogger(__name__)


class Ligand(BaseLigand):
    """
    General small molecule object supporting RDKit, OpenForceField and OpenEye toolkits. During
    initialization attributes are stored as given:

    >>> ligand = Ligand(smiles='CCC')

    Use one of the following attributes to get a molecular representation of your favorite toolkit:

    >>> rdkit_mol = Ligand.rdkit_mol
    >>> openff_mol = Ligand.openff_mol
    >>> openeye_mol = Ligand.openeye_mol

    Parameters
    ---------
    name: str, default=''
        The name of the small molecule.
    smiles: str, default=''
        A SMILES string.
    sdf_path: str or Path, default=''
        The path to the small molecule file in SDF format.
    rdkit_mol: rdkit.Chem.Mol or None, default=None
        An RDKit molecule.
    openff_mol: openff.toolkit.topology.Molecule or None, default=None
        An OpenForceField molecule.
    openeye_mol: oechem.OEGraphMol or None, default=None
        An OpenEye molecule.
    metadata: dict or None, defualt=None
        Additional metadata.
    """
    def __init__(
            self, name="", smiles="", sdf_path="", rdkit_mol=None, openff_mol=None, openeye_mol=None,
            metadata=None, *args, **kwargs
    ):
        super().__init__(name=name, metadata=metadata, *args, **kwargs)
        self.smiles = smiles
        self.sdf_path = sdf_path
        self._rdkit_mol = rdkit_mol
        self._openff_mol = openff_mol
        self._openeye_mol = openeye_mol

    @property
    def rdkit_mol(self):
        return self._rdkit_mol

    @rdkit_mol.setter
    def rdkit_mol(self, new_value):
        self._rdkit_mol = new_value

    @rdkit_mol.getter
    def rdkit_mol(self):
        if not self._rdkit_mol:
            if self.sdf_path:
                from rdkit import Chem
                supplier = Chem.SDMolSupplier(self.sdf_path)
                self._rdkit_mol = next(supplier)
            elif self.smiles:
                from rdkit import Chem
                self._rdkit_mol = Chem.MolFromSmiles(self.smiles)
            elif self._openff_mol:
                self._rdkit_mol = self._openff_mol.to_rdkit()
            elif self._openeye_mol:
                from openff.toolkit.topology import Molecule
                self._rdkit_mol = Molecule.from_openeye(self._openeye_mol).to_rdkit()
            else:
                raise ValueError(
                    "To allow access to RDKit molecules, the `Ligand` object needs to be "
                    "initialized with one of the following attributes:\nsmiles\nsdf_path\n"
                    "rdkit_mol\nopenff_mol\nopeneye_mol"
                )
        return self._rdkit_mol

    @property
    def openff_mol(self):
        return self._openff_mol

    @openff_mol.setter
    def openff_mol(self, new_value):
        self._openff_mol = new_value

    @openff_mol.getter
    def openff_mol(self):
        if not self._openff_mol:
            from openff.toolkit.topology import Molecule
            if self.sdf_path:
                self._openff_mol = Molecule.from_file(self.sdf_path)
            elif self.smiles:
                self._openff_mol = Molecule.from_smiles(self.smiles, allow_undefined_stereo=True)
            elif self._rdkit_mol:
                self._openff_mol = Molecule.from_rdkit(self._rdkit_mol)
            elif self._openeye_mol:
                self._openff_mol = Molecule.from_openeye(self._openeye_mol)
            else:
                raise ValueError(
                    "To allow access to OpenForceField molecules, the `Ligand` object needs to be "
                    "initialized with one of the following attributes:\nsmiles\nsdf_path\n"
                    "rdkit_mol\nopenff_mol\nopeneye_mol\n"
                )
        return self._openff_mol

    @property
    def openeye_mol(self):
        return self._openeye_mol

    @openeye_mol.setter
    def openeye_mol(self, new_value):
        self._openeye_mol = new_value

    @openeye_mol.getter
    def openeye_mol(self):
        if not self._openeye_mol:
            if self.sdf_path:
                from ..modeling.OEModeling import read_molecules
                self._openeye_mol = read_molecules(self.sdf_path)[0]
            elif self.smiles:
                from ..modeling.OEModeling import read_smiles
                self._openeye_mol = read_smiles(self.smiles)
            elif self._openff_mol:
                self._openeye_mol = self._openff_mol.to_openeye()
            elif self._rdkit_mol:
                from openff.toolkit.topology import Molecule
                self._openeye_mol = Molecule.from_rdkit(self._rdkit_mol).to_openeye()
            else:
                raise ValueError(
                    "To allow access to OpenEye molecules, the `Ligand` object needs to be "
                    "initialized with one of the following attributes:\nsmiles\nsdf_path\n"
                    "rdkit_mol\nopenff_mol\nopeneye_mol"
                )
        return self._openeye_mol

    def get_canonical_smiles(self, toolkit="rdkit") -> str:
        """
        Generate a canonical isomeric SMILES string.

        Parameters
        ----------
        toolkit: str, default='rdkit'
            The toolkit to use for generating the canonical SMILES ('rdkit', 'openff', 'openeye').

        Returns
        -------
            : str
            The canonical isomeric SMILES string.
        """
        if toolkit == "rdkit":
            from rdkit import Chem
            return Chem.MolToSmiles(self.rdkit_mol, allHsExplicit=False)
        elif toolkit == "openff":
            return self.openff_mol.to_smiles(explicit_hydrogens=False)
        elif toolkit == "openeye":
            from openeye import oechem
            return oechem.OEMolToSmiles(self.openeye_mol)
        else:
            raise ValueError(
                "Provide a supported toolkit to export the SMILES string, i.e. 'rdkit', 'openff' or "
                "'openeye'."
            )
