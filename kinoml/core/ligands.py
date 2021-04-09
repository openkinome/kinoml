"""
``MolecularComponent`` objects that represent ligand-like entities.
"""

import logging

import rdkit
from openff.toolkit.topology import Molecule as _OpenForceFieldMolecule

from .components import BaseLigand
from ..utils import download_file

logger = logging.getLogger(__name__)


class FileLigand(BaseLigand):
    """
    Docstring pending
    """

    def __init__(self, path, metadata=None, name="", *args, **kwargs):
        super().__init__(name=name, metadata=metadata, *args, **kwargs)
        if str(path).startswith("http"):
            from appdirs import user_cache_dir

            # TODO: where to save, how to name
            self.path = f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
            download_file(path, self.path)
        else:
            self.path = path


class PDBLigand(FileLigand):
    """
    Docstring pending
    """

    def __init__(self, pdb_id, path, metadata=None, name="", *args, **kwargs):
        super().__init__(path, metadata=metadata, name=name, *args, **kwargs)
        from appdirs import user_cache_dir

        self.pdb_id = pdb_id
        self.path = f"{user_cache_dir()}/{self.name}.sdf"  # <- SDF? Isn't this a PDB?
        download_file(f"https://files.rcsb.org/ligands/view/{pdb_id}_ideal.sdf", self.path)


class OpenForceFieldLigand(BaseLigand, _OpenForceFieldMolecule):

    """
    Small molecule object based on the OpenForceField toolkit.

    Instantiation usually happens through a ``.from_xxxx()``
    class method.

    Parameters
    ----------
    metadata : dict
        Metadata for this molecule, like provenance information
        or the original SMILES string used to instantiate the object.

    Examples
    --------
    >>> ligand = OpenForceFieldLigand.from_smiles("CCCC")
    """

    def __init__(self, metadata=None, name="", *args, **kwargs):
        _OpenForceFieldMolecule.__init__(self, *args, **kwargs)
        BaseLigand.__init__(self, name=name, metadata=metadata)

    @classmethod
    def from_smiles(
        cls, smiles, name=None, allow_undefined_stereo=True, **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Same as `openff.toolkit.topology.Molecule`, but adding
        information about the original SMILES to ``.metadata`` dict.

        Parameters
        ----------
        smiles : str
            SMILES representation of the ligand. This string will
            be stored in the ``metadata`` attribute under the
            ``smiles`` key.
        name : str, optional
            An easily identifiable name for the molecule. If not given,
            ``smiles`` is used.
        """
        self = super().from_smiles(smiles, allow_undefined_stereo=allow_undefined_stereo, **kwargs)
        if name is None:
            name = smiles
        super().__init__(self, name=name, metadata={"smiles": smiles})
        return self

    def to_dict(self):
        """
        Dict representation of the Molecule, including the ``metadata``
        dictionary.
        """
        d = super().to_dict()
        d["metadata"] = self.metadata.copy()
        return d

    def _initialize_from_dict(self, molecule_dict):
        """
        Same as Molecule's method, but including the ``metadata`` dict.
        """
        super()._initialize_from_dict(molecule_dict)
        self.metadata = molecule_dict["metadata"].copy()


# Alias OpenForceFieldLigand to Ligand
Ligand = OpenForceFieldLigand


class OpenForceFieldLikeLigand(BaseLigand):
    """
    Ligand-like object that implements the bits of the
    OpenForceField API we use more commonly.

    The attributes of the wrapped object are forwarded
    to ``self._molecule`` via ``__getattr__`` to provide
    most of the native behaviour.

    This is only the base class; use concrete subclasses
    for full functionality.

    Parameters
    ----------
    molecule : object, depends on subclass
        The molecular object to be wrapped, under ``._molecule``.
    metadata : dict, optional
        Metadata dictionary
    name : str, optional
        Easily identifiable name for this ligand
    """

    def __init__(self, molecule, metadata=None, name="", *args, **kwargs):
        super().__init__(name=name, metadata=metadata)
        self._molecule = molecule

    def __getattr__(self, attr):
        """
        Forward attribute access to the wrapped ``._molecule`` object
        """
        if attr in {"__getstate__", "__setstate__"}:
            return super().__getattr__(self, attr)
        return getattr(self._molecule, attr)

    @classmethod
    def from_smiles(cls, smiles, name=None, **kwargs):
        """
        Create object from SMILES
        """
        raise NotImplementedError("Use ``OpenForceFieldLigand`` or implement API in a subclass")

    def to_rdkit(self) -> rdkit.Chem.Mol:
        """
        Export Molecule to RDKit ``Mol``

        Returns
        -------
        rdkit.Chem.Mol
        """
        raise NotImplementedError("Use ``OpenForceFieldLigand`` or implement API in a subclass")

    def to_smiles(self) -> str:
        """
        Export Molecule to (canonical) SMILES string.

        Returns
        -------
        str
        """
        raise NotImplementedError("Use ``OpenForceFieldLigand`` or implement API in a subclass")


class RDKitLigand(OpenForceFieldLikeLigand):

    """
    Wrapper for RDKit molecules using some parts of the OpenForceField API

    Note
    ----
    TODO: Implement other parts of the OFF Molecule API
    """

    @classmethod
    def from_smiles(
        cls, smiles: str, name: str = None, **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Create an RDKitLigand instance from a SMILES string

        Parameters
        ----------
        smiles : str
            SMILES sequence encoding the required molecule
        name : str, optional
            Identifier for the molecule. If not given, ``smiles``
            will be used.

        Note
        ----
        The ``metadata`` dictionary will be populated with a
        ``smiles`` entry containing the input ``smiles`` string.
        """
        from rdkit.Chem import MolFromSmiles

        molecule = MolFromSmiles(smiles)
        if name is None:
            name = smiles
        return cls(molecule, name=name, metadata={"smiles": smiles})

    def to_rdkit(self) -> rdkit.Chem.Mol:
        """
        Return the underlying RDKit ``Mol`` object, with no further
        modifications.

        Returns
        -------
        rdkit.Chem.Mol
        """
        return self._molecule

    def to_smiles(self) -> str:
        """
        Return canonicalized SMILES, as provided by RDKit.

        Note
        ----
        More info: https://www.rdkit.org/docs/GettingStartedInPython.html#writing-molecules
        """
        from rdkit.Chem import MolToSmiles

        return MolToSmiles(self._molecule)


class SmilesLigand(OpenForceFieldLikeLigand):
    """
    Wrap a SMILES string in an OpenForceField-like API.

    The underlying ``._molecule`` is just the SMILES string,
    with no preprocessing.
    """

    @classmethod
    def from_smiles(
        cls, smiles: str, name: str = None, **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Initialize a SmilesLigand object using ``smiles`` as input.

        Parameters
        ----------
        smiles : str
            The SMILES string to wrap
        name : str, optional
            Identifier for this molecule. If not given, ``smiles``
            will be used

        Note
        ----
        The ``metadata`` dictionary will also contain a ``smiles``
        key containing the input SMILES, for API compatibility reasons.
        """
        return cls(smiles, name=name or smiles, metadata={"smiles": smiles})

    def to_rdkit(self) -> rdkit.Chem.Mol:
        """
        Export this SMILES string as an RDKit ``Mol``.

        Returns
        -------
        rdkit.Chem.Mol
        """
        return RDKitLigand.from_smiles(self._molecule).to_rdkit()

    def to_smiles(self) -> str:
        """
        Create an RDKit ``Mol`` and export it as canonical SMILES
        representation. If you want the RAW smiles, use
        ``.metadata["smiles"]``.

        Returns
        -------
        str
            Canonical SMILES
        """
        return RDKitLigand.from_smiles(self._molecule).to_smiles()