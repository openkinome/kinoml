"""
``MolecularComponent`` objects that represent protein-like entities.
"""
import logging

from .components import BaseProtein
from .sequences import AminoAcidSequence

logger = logging.getLogger(__name__)


class Protein(BaseProtein, AminoAcidSequence):
    """
    General protein object supporting MDAnalysis and OpenEye toolkits.
    """

    def __init__(
            self,
            pdb_id="",
            molecule=None,
            toolkit="OpenEye",
            name="",
            sequence="",
            uniprot_id="",
            ncbi_id="",
            metadata=None,
            **kwargs
    ):
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
        self._molecule = molecule
        if toolkit not in ["OpenEye", "MDAnalysis"]:
            raise AttributeError(
                f"Only 'MDAnalysis' and 'OpenEye' are supported, you provided '{toolkit}'."
            )
        self.toolkit = toolkit

    @property
    def pdb_id(self):
        return self._pdb_id

    @pdb_id.setter
    def pdb_id(self, new_value):
        raise AttributeError(
            f"Do not modify pdb_id after instantiation, create a new {self.__class__.__name__} "
            f"object instead."
        )

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, new_value):
        self._molecule = new_value

    @molecule.getter
    def molecule(self):
        if not self._molecule and self.pdb_id:
            from ..databases.pdb import download_pdb_structure
            file_path = download_pdb_structure(self.pdb_id)
            if self.toolkit == "OpenEye":
                from ..modeling.OEModeling import read_molecules
                self._molecule = read_molecules(file_path)[0]
            elif self.toolkit == "MDAnalysis":
                from ..modeling.MDAnalysisModeling import read_molecule
                self._molecule = read_molecule(file_path)
            if self.metadata is None:
                self.metadata = {"pdb_id": self.pdb_id}
            else:
                self.metadata.update({"smiles": self.pdb_id})
        return self._molecule

    @classmethod
    def from_file(cls, file_path, name="", toolkit="OpenEye"):
        if toolkit == "OpenEye":
            from ..modeling.OEModeling import read_molecules
            molecule = read_molecules(file_path)[0]
        else:
            from ..modeling.MDAnalysisModeling import read_molecule
            molecule = read_molecule(file_path)

        return cls(molecule=molecule, name=name, toolkit=toolkit, metadata={"file_path": file_path})

    @classmethod
    def from_pdb(cls, pdb_id, name="", toolkit="OpenEye"):
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
    Kinase object with access to KLIFS and supporting MDAnalysis and OpenEye toolkits.
    """

    def __init__(
            self,
            pdb_id="",
            molecule=None,
            toolkit="OpenEye",
            name="",
            sequence="",
            uniprot_id="",
            structure_klifs_id=None,
            kinase_klifs_id=None,
            kinase_klifs_sequence=None,
            structure_klifs_sequence=None,
            structure_klifs_residues=None,
            metadata=None,
            **kwargs
    ):
        super().__init__(
            pdb_id=pdb_id,
            molecule=molecule,
            toolkit=toolkit,
            name=name,
            sequence=sequence,
            uniprot_id=uniprot_id,
            metadata=metadata,
            **kwargs
        )
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
        return self._kinase_klifs_sequence

    @kinase_klifs_sequence.setter
    def kinase_klifs_sequence(self, new_value):
        self._kinase_klifs_sequence = new_value

    @kinase_klifs_sequence.getter
    def kinase_klifs_sequence(self):
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
                        "To allow access to the kinase KLIFS sequence, the `Kinase` object needs "
                        "to be initialized with one of the following attributes:"
                        "\nkinase_klifs_sequence\nkinase_klifs_id\nstructure_klifs_id"
                        "\nuniprot_id"
                    )
            kinase_details = remote.kinases.by_kinase_klifs_id(self.kinase_klifs_id)
            self._kinase_klifs_sequence = kinase_details["kinase.pocket"].values[0]
        return self._kinase_klifs_sequence

    @property
    def structure_klifs_sequence(self):
        return self._structure_klifs_sequence

    @structure_klifs_sequence.setter
    def structure_klifs_sequence(self, new_value):
        self._structure_klifs_sequence = new_value

    @structure_klifs_sequence.getter
    def structure_klifs_sequence(self):
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
                    "To allow access to the structure KLIFS sequence, the `Kinase` object needs "
                    "to be initialized with one of the following attributes:"
                    "\nstructure_klifs_sequence\nstructure_klifs_id"
                )
        return self._structure_klifs_sequence

    @property
    def structure_klifs_residues(self):
        return self._structure_klifs_residues

    @structure_klifs_residues.setter
    def structure_klifs_residues(self, new_value):
        self._structure_klifs_residues = new_value

    @structure_klifs_residues.getter
    def structure_klifs_residues(self):
        if not self._structure_klifs_residues:
            if self.structure_klifs_id:
                from opencadd.databases.klifs import setup_remote

                remote = setup_remote()
                self._structure_klifs_residues = remote.pockets.by_structure_klifs_id(
                    self.structure_klifs_id
                )
            else:
                raise ValueError(
                    "To allow access to structure KLIFS residues, the `Kinase` object needs to "
                    "be initialized with one of the following attributes:"
                    "\nstructure_klifs_residues\nstructure_klifs_id"
                )

        return self._structure_klifs_residues
