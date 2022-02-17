"""
``MolecularComponent`` objects that represent protein-like entities.
"""
import logging
from pathlib import Path
from typing import Union

from MDAnalysis.core.universe import Universe, AtomGroup
from openeye import oechem
import pandas as pd

from .components import BaseProtein
from .sequences import AminoAcidSequence


logger = logging.getLogger(__name__)


class Protein(BaseProtein, AminoAcidSequence):
    """
    Create a new Protein object. A molecular representation is accessible via the molecule attribute.

    Examples
    --------

    Create a protein from file with OpenEye toolkit molecular representation:

    >>> protein = Protein.from_file("data/proteins/4f8o.pdb", name="4f8o")

    Create a protein from file with MDAnalysis toolkit molecular representation:

    >>> protein = Protein.from_file("data/proteins/4f8o.pdb", name="4f8o", toolkit="MDAnalysis")

    Create a protein from an OpenEye molecule:

    >>> from kinoml.modeling.OEModeling import read_molecules
    >>> molecule = read_molecules("data/proteins/4f8o.pdb")[0]
    >>> protein = Protein(molecule=molecule, name="4f8o")

    Create a protein from PDB ID:

    >>> protein = Protein.from_pdb("4f8o")

    Create a protein from PDB ID with lazy instantiation:

    >>> protein = Protein(pdb_id="4f8o")

    Create a protein from PDB ID with lazy instantiation and get access to the complete wildtype
    amino acid sequence via providing a UniProt ID:

    >>> protein = Protein(pdb_id="4f8o", uniprot_id="P31522")
    >>> protein.sequence

    """
    def __init__(
            self,
            pdb_id: str = "",
            molecule: Union[oechem.OEMol, oechem.OEGraphMol, Universe, None] = None,
            toolkit: str = "OpenEye",
            name: str = "",
            sequence: str = "",
            uniprot_id: str = "",
            ncbi_id: str = "",
            metadata: Union[dict, None] = None,
            **kwargs
    ):
        """
        Create a new Protein object. Lazy instantiation is possible via the pdb_id parameter.

        Parameters
        ----------
        pdb_id: str, default=""
            The PDB ID of the protein.
        molecule: Universe or AtomGroup or oechem.OEMol or oechem.OEGraphMol or None, default=None
            A molecular representation of the protein via OpenEye or MDAnalysis.
        toolkit: str, default="OpenEye"
            The toolkit to use for molecular representation ("MDAnalysis" or "OpenEye").
        name: str, default=""
            The name of the protein.
        sequence: str, default=""
            The amino acid sequence of the protein.
        uniprot_id: str, default=""
            The UniProt ID of the protein.
        ncbi_id: str, default=""
            The NCBI ID of the protein.
        metadata: dict or None, default=None
            Additional metadata of the needed for e.g. featurizers or provenance.
        """
        BaseProtein.__init__(self)
        AminoAcidSequence.__init__(
            self,
            name=name,
            sequence=sequence,
            uniprot_id=uniprot_id,
            ncbi_id=ncbi_id,
            metadata=metadata,
            **kwargs
        )
        self._pdb_id = pdb_id
        if toolkit not in ["OpenEye", "MDAnalysis"]:
            raise AttributeError(
                f"Only 'MDAnalysis' and 'OpenEye' are supported, you provided '{toolkit}'."
            )
        if molecule:
            if isinstance(molecule, (oechem.OEMol, oechem.OEGraphMol)):
                toolkit = "OpenEye"
            elif isinstance(molecule, (Universe, AtomGroup)):
                toolkit = "MDAnalysis"
            else:
                raise ValueError(f"{type(molecule)} is not a supported type for molecule.")
        self._molecule = molecule
        self.toolkit = toolkit

    @property
    def pdb_id(self):
        """Decorate pdb_id to modify setter."""
        return self._pdb_id

    @pdb_id.setter
    def pdb_id(self, new_value):
        """
        Prevent setting a new pdb_id after instantiation.

        Raises
        ------
        AttributeError
            Do not modify pdb_id after instantiation, create a new Protein object instead.
        """
        raise AttributeError(
            f"Do not modify pdb_id after instantiation, create a new {self.__class__.__name__} "
            f"object instead."
        )

    @property
    def molecule(self):
        """Decorate molecule to modify setter and getter."""
        return self._molecule

    @molecule.setter
    def molecule(self, new_value):
        """
        Store a new value for molecule in the _molecule attribute.

        Parameters
        ----------
        new_value: Universe or AtomGroup or oechem.OEMol or oechem.OEGraphMol or None
            A new molecular representation of the protein via OpenEye or MDAnalysis.
        """
        self._molecule = new_value

    @molecule.getter
    def molecule(self):
        """
        Get the _molecule attribute. If the pdb_id attribute is given and _molecule is None, a
        new molecule representation will be created from the given pdb_id, e.g. in case of lazy
        instantiation. The toolkit being used depends on the toolkit attribute.

        Returns
        ------
        : Universe or AtomGroup or oechem.OEMol or oechem.OEGraphMol or None
            The molecular representation of the protein.
        """
        if not self._molecule and self.pdb_id:
            from ..databases.pdb import download_pdb_structure
            file_path = download_pdb_structure(self.pdb_id)
            if self.toolkit == "OpenEye":
                from ..modeling.OEModeling import read_molecules
                self._molecule = read_molecules(file_path)[0]
            elif self.toolkit == "MDAnalysis":
                from ..modeling.MDAnalysisModeling import read_molecule
                self._molecule = read_molecule(file_path)
            if not self.name:
                self.name = self.pdb_id
            if self.metadata is None:
                self.metadata = {"pdb_id": self.pdb_id}
            else:
                self.metadata.update({"pdb_id": self.pdb_id})
        return self._molecule

    @classmethod
    def from_file(cls, file_path: Union[Path, str], name: str = "", toolkit: str = "OpenEye"):
        """
        Create a Protein from file.

        Parameters
        ----------
        file_path: pathlib.Path or str
            The path to the molecular file. Supported formats depend on the toolkit being used.
        name: str, default=""
            The name of the protein.
        toolkit: str, default="OpenEye"
            The toolkit to use for molecular representation.
        """
        if toolkit == "OpenEye":
            from ..modeling.OEModeling import read_molecules
            molecule = read_molecules(file_path)[0]
        else:
            from ..modeling.MDAnalysisModeling import read_molecule
            molecule = read_molecule(file_path)

        return cls(
            molecule=molecule,
            name=name,
            toolkit=toolkit,
            metadata={"file_path": file_path},
        )

    @classmethod
    def from_pdb(cls, pdb_id: str, name: str = "", toolkit: str = "OpenEye"):
        """
        Create a Protein from file.

        Parameters
        ----------
        pdb_id: str
            The PDB ID of the protein structure of interest.
        name: str, default=""
            The name of the protein.
        toolkit: str, default="OpenEye"
            The toolkit to use for molecular representation.
        """
        from ..databases.pdb import download_pdb_structure
        file_path = download_pdb_structure(pdb_id)
        if toolkit == "OpenEye":
            from ..modeling.OEModeling import read_molecules
            molecule = read_molecules(file_path)[0]
        else:
            from ..modeling.MDAnalysisModeling import read_molecule
            molecule = read_molecule(file_path)
        if not name:
            name = pdb_id
        return cls(molecule=molecule, name=name, toolkit=toolkit, metadata={"pdb_id": pdb_id})


class KLIFSKinase(Protein):
    """
    Create a new KLIFSKinase object. A molecular representation is accessible via the molecule
    attribute. Allows access to the sequence and residues of the KLIFS binding pocket.

    Examples
    --------

    Create a KLIFS kinase from PDB ID with lazy instantiation:

    >>> kinase = KLIFSKinase(pdb_id="4yne")

    Create a KLIFS kinase from PDB ID with lazy instantiation and gain access to the wildtype
    KLIFS pocket sequence via providing a UniProt ID:

    >>> kinase = KLIFSKinase(pdb_id="4yne", uniprot_id="P04629")
    >>> kinase.kinase_klifs_sequence()

    Create a KLIFS kinase from PDB ID with lazy instantiation and gain access to the wildtype
    KLIFS pocket sequence via providing a KLIFS specifc kinase ID:

    >>> kinase = KLIFSKinase(pdb_id="4yne", kinase_klifs_id=480)
    >>> kinase.kinase_klifs_sequence()  # wildtype, does not need to match the given PDB structure

    Create a KLIFS kinase from PDB ID with lazy instantiation and gain access to the KLIFS pocket
    sequence and residues of the structure via providing a KLIFS specifc structure ID:

    >>> kinase = KLIFSKinase(pdb_id="4yne", structure_klifs_id=3620)
    >>> kinase.kinase_klifs_sequence()  # wildtype, does not need to match the given PDB structure
    >>> kinase.structure_klifs_sequence()  # specific to the structure
    >>> kinase.structure_klifs_residues()  # specific to the structure

    """

    def __init__(
            self,
            pdb_id: str = "",
            molecule: Union[oechem.OEMol, oechem.OEGraphMol, Universe, None] = None,
            toolkit: str = "OpenEye",
            name: str = "",
            sequence: str = "",
            uniprot_id: str = "",
            ncbi_id: str = "",
            structure_klifs_id: Union[int, None] = None,
            kinase_klifs_id: Union[int, None] = None,
            kinase_klifs_sequence: str = "",
            structure_klifs_sequence: str = "",
            structure_klifs_residues: Union[pd.DataFrame, None] = None,
            metadata: Union[dict, None] = None,
            **kwargs
    ):
        super().__init__(
            pdb_id=pdb_id,
            molecule=molecule,
            toolkit=toolkit,
            name=name,
            sequence=sequence,
            uniprot_id=uniprot_id,
            ncbi_id=ncbi_id,
            metadata=metadata,
            **kwargs
        )
        """
        Create a new KLIFSKinase object. Lazy instantiation is possible via the pdb_id parameter.

        Parameters
        ----------
        pdb_id: str, default=""
            The PDB ID of the protein.
        molecule: Universe or AtomGroup or oechem.OEMol or oechem.OEGraphMol or None, default=None
            A molecular representation of the protein via OpenEye or MDAnalysis.
        toolkit: str, default="OpenEye"
            The toolkit to use for molecular representation ("MDAnalysis" or "OpenEye").
        name: str, default=""
            The name of the protein.
        sequence: str, default=""
            The amino acid sequence of the protein.
        uniprot_id: str, default=""
            The UniProt ID of the protein.
        ncbi_id: str, default=""
            The NCBI ID of the protein.
        structure_klifs_id: int or None, default=None
            The structure KLIFS ID.
        kinase_klifs_id: int or None, default=None
            The kinase KLIFS ID.
        kinase_klifs_sequence: str, default=""
            The widtype kinase KLIFS binding pocket sequence. Can be inferred from the uniprot_id, 
            structure_klifs_id or kinase_klifs_id.
        structure_klifs_sequence: str, default=""
            The KLIFS binding pocket sequence of the given structure. Can be inferred from the 
            structure_klifs_id.
        structure_klifs_residues: pd.DataFrame or None, default=None
            The structure-specific KLIFS residues formatted like by opencadd.databases.klifs.
        metadata: dict or None, default=None
            Additional metadata of the needed for e.g. featurizers or provenance.
        """
        self.structure_klifs_id = structure_klifs_id
        self.kinase_klifs_id = kinase_klifs_id
        self._kinase_klifs_sequence = kinase_klifs_sequence
        self._structure_klifs_sequence = structure_klifs_sequence
        self._structure_klifs_residues = structure_klifs_residues

    def _query_sequence_sources(self):
        """
        Query available sources for sequence details. Add additional methods below to allow
        fetching from other sources.
        """
        if self.uniprot_id:
            self._query_uniprot()
        elif self.ncbi_id:
            self._query_ncbi()
        elif self.structure_klifs_id or self.kinase_klifs_id:
            self._query_klifs()

    def _query_klifs(self):
        """Query KLIFS for the Uniprot ID, which allows fetching of the sequence."""
        from opencadd.databases.klifs import setup_remote

        remote = setup_remote()

        if self.structure_klifs_id and not self.kinase_klifs_id:
            structure_details = remote.structures.by_structure_klifs_id(self.structure_klifs_id)
            self.kinase_klifs_id = structure_details["kinase.klifs_id"].iloc[0]

        if self.kinase_klifs_id:
            kinase_details = remote.kinases.by_kinase_klifs_id(self.kinase_klifs_id)
            self.uniprot_id = kinase_details["kinase.uniprot"].iloc[0]

        self._query_uniprot()

    @property
    def kinase_klifs_sequence(self):
        """Decorate kinase_klifs_sequence to modify setter and getter."""
        return self._kinase_klifs_sequence

    @kinase_klifs_sequence.setter
    def kinase_klifs_sequence(self, new_value):
        """
        Store a new value for kinase_klifs_sequence in the _kinase_klifs_sequence attribute.

        Parameters
        ----------
        new_value: str
            A new amino acid sequence of the KLIFS kinase pocket.
        """
        self._kinase_klifs_sequence = new_value

    @kinase_klifs_sequence.getter
    def kinase_klifs_sequence(self):
        """
        Get the _kinase_klifs_sequence attribute. Query KLIFS for the respective sequence if
        _kinase_klifs_sequence is an empty string.

        Returns
        ------
        : str
            The kinase KLIFS binding pocket sequence.

        Raises
        ------
        ValueError
            To allow access to the kinase KLIFS sequence, the `KLIFSKinase` object needs to be
            initialized with one of the following attributes:
            kinase_klifs_sequence
            kinase_klifs_id
            structure_klifs_id
            uniprot_id
        """
        if not self._kinase_klifs_sequence:
            from opencadd.databases.klifs import setup_remote
            remote = setup_remote()
            if not self.kinase_klifs_id:
                if self.structure_klifs_id:
                    structure_details = remote.structures.by_structure_klifs_id(
                        self.structure_klifs_id
                    )
                    self.kinase_klifs_id = structure_details["kinase.klifs_id"].iloc[0]
                elif self.uniprot_id:
                    all_kinases = remote.kinases.all_kinases()
                    kinases = all_kinases[all_kinases["kinase.uniprot"] == self.uniprot_id]
                    if len(kinases) > 0:
                        self.kinase_klifs_id = kinases.iloc[0]["kinase.klifs_id"]
                    else:
                        raise ValueError(
                            f"Could not find a kinase in KLIFS for uniprot ID '{self.uniprot_id}'."
                        )
                else:
                    raise ValueError(
                        "To allow access to the kinase KLIFS sequence, the `KLIFSKinase` object "
                        "needs to be initialized with one of the following attributes:"
                        "\nkinase_klifs_sequence\nkinase_klifs_id\nstructure_klifs_id"
                        "\nuniprot_id"
                    )
            kinase_details = remote.kinases.by_kinase_klifs_id(self.kinase_klifs_id)
            self._kinase_klifs_sequence = kinase_details["kinase.pocket"].values[0]
        return self._kinase_klifs_sequence

    @property
    def structure_klifs_sequence(self):
        """Decorate structure_klifs_sequence to modify setter and getter."""
        return self._structure_klifs_sequence

    @structure_klifs_sequence.setter
    def structure_klifs_sequence(self, new_value):
        """
        Store a new value for structure_klifs_sequence in the _structure_klifs_sequence attribute.

        Parameters
        ----------
        new_value: str
            A new amino acid sequence of the structure-specific KLIFS kinase pocket.
        """
        self._structure_klifs_sequence = new_value

    @structure_klifs_sequence.getter
    def structure_klifs_sequence(self):
        """
        Get the _structure_klifs_sequence attribute. Query KLIFS for the respective sequence if
        _structure_klifs_sequence is an empty string.

        Returns
        ------
        : str
            The structure-specific KLIFS binding pocket sequence.

        Raises
        ------
        ValueError
            To allow access to the structure KLIFS sequence, the `KLIFSKinase` object needs to be
            initialized with one of the following attributes:
            structure_klifs_sequence
            structure_klifs_id
        """
        if not self._structure_klifs_sequence:
            if self.structure_klifs_id:
                from opencadd.databases.klifs import setup_remote

                remote = setup_remote()
                structure_details = remote.structures.by_structure_klifs_id(
                    self.structure_klifs_id
                )
                self._structure_klifs_sequence = structure_details["structure.pocket"].values[0]
            else:
                raise ValueError(
                    "To allow access to the structure KLIFS sequence, the `KLIFSKinase` object "
                    "needs to be initialized with one of the following attributes:"
                    "\nstructure_klifs_sequence\nstructure_klifs_id"
                )
        return self._structure_klifs_sequence

    @property
    def structure_klifs_residues(self):
        """Decorate structure_klifs_residues to modify setter and getter."""
        return self._structure_klifs_residues

    @structure_klifs_residues.setter
    def structure_klifs_residues(self, new_value):
        """
        Store a new value for structure_klifs_residues in the _structure_klifs_residues attribute.

        Parameters
        ----------
        new_value: pd.DataFrame or None
            The new structure-specific KLIFS residues formatted like by opencadd.databases.klifs.
        """
        self._structure_klifs_residues = new_value

    @structure_klifs_residues.getter
    def structure_klifs_residues(self):
        """
        Get the _structure_klifs_residues attribute. Query KLIFS for the respective residues if
        _structure_klifs_residues is None.

        Returns
        ------
        : pd.DataFrame or None
            The structure-specific KLIFS residues formatted like by opencadd.databases.klifs.

        Raises
        ------
        ValueError
            To allow access to structure KLIFS residues, the `KLIFSKinase` object needs to be
            initialized with one of the following attributes:
            structure_klifs_residues
            structure_klifs_id
        """
        if self._structure_klifs_residues is None:
            if self.structure_klifs_id:
                from opencadd.databases.klifs import setup_remote

                remote = setup_remote()
                self._structure_klifs_residues = remote.pockets.by_structure_klifs_id(
                    self.structure_klifs_id
                )
            else:
                raise ValueError(
                    "To allow access to structure KLIFS residues, the `KLIFSKinase` object needs to "
                    "be initialized with one of the following attributes:"
                    "\nstructure_klifs_residues\nstructure_klifs_id"
                )

        return self._structure_klifs_residues
