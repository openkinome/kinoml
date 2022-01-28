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

    Use one of the following methods to get a molecular representation of your favorite toolkit:

    >>> rdkit_mol = Ligand.to_rdkit()
    >>> openff_mol = Ligand.to_openff()
    >>> openeye_mol = Ligand.to_openeye()

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
        self._smiles = smiles
        self._sdf_path = sdf_path
        self._rdkit_mol = rdkit_mol
        self._openff_mol = openff_mol
        self._openeye_mol = openeye_mol

    def to_rdkit(self):
        """
        Export an RDKit molecule.

        Returns
        -------
            : rdkit.Chem.Mol
        """
        if not self._rdkit_mol:
            if self._sdf_path:
                from rdkit import Chem
                supplier = Chem.SDMolSupplier(self._sdf_path)
                self._rdkit_mol = next(supplier)
            elif self._smiles:
                from rdkit import Chem
                self._rdkit_mol = Chem.MolFromSmiles(self._smiles)
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

    def to_openff(self):
        """
        Export an OpenForceField molecule.

        Returns
        -------
            : openff.toolkit.topology.Molecule
        """
        if not self._openff_mol:
            from openff.toolkit.topology import Molecule
            if self._sdf_path:
                self._openff_mol = Molecule.from_file(self._sdf_path)
            elif self._smiles:
                self._openff_mol = Molecule.from_smiles(self._smiles, allow_undefined_stereo=True)
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

    def to_openeye(self):
        """
        Export an OpenEye molecule.

        Returns
        -------
            : oechem.OEGraphMol
        """
        if not self._openeye_mol:
            if self._sdf_path:
                from ..modeling.OEModeling import read_molecules
                self._openeye_mol = read_molecules(self._sdf_path)[0]
            elif self._smiles:
                from ..modeling.OEModeling import read_smiles
                self._openeye_mol = read_smiles(self._smiles)
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

    def to_smiles(self, toolkit="rdkit") -> str:
        """
        Export Molecule to a canonical isomeric SMILES string.

        Parameters
        ----------
        toolkit: str, default='rdkit'
            The toolkit to use for generating the canonical smiles ('rdkit', 'openff', 'openeye').

        Returns
        -------
            : str
            The canonical isomeric SMILES string.
        """
        if toolkit == "rdkit":
            from rdkit import Chem
            if not self._rdkit_mol:
                self.to_rdkit()
            return Chem.MolToSmiles(self._rdkit_mol, allHsExplicit=False)
        elif toolkit == "openff":
            if not self._openff_mol:
                self.to_openff()
            return self._openff_mol.to_smiles(explicit_hydrogens=False)
        elif toolkit == "openeye":
            from openeye import oechem
            if not self._openeye_mol:
                self.to_openeye()
            return oechem.OEMolToSmiles(self._openeye_mol)
        else:
            raise ValueError(
                "Provide a supported toolkit to export the SMILES string, i.e. 'rdkit', 'openff' or "
                "'openeye'."
            )
