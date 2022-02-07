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
            file_path="",
            chain_id="",
            alternate_location="",
            ligand_name="",
            mda_mol=None,
            openeye_mol=None,
            name="",
            sequence="",
            uniprot_id=None,
            metadata=None,
            **kwargs
    ):
        BaseProtein.__init__(self)
        AminoAcidSequence.__init__(
            self,
            name=name,
            sequence=sequence,
            uniprot_id=uniprot_id,
            # ncbi
            metadata=metadata,
            **kwargs
        )
        self.pdb_id = pdb_id  # only pdb_id is lazy
        self.file_path = file_path
        self.chain_id = chain_id
        self.alternate_location = alternate_location
        self.ligand_name = ligand_name
        self._mda_mol = mda_mol  # write mdanaylsis
        self._openeye_mol = openeye_mol  # to_openff, to_mdanalysis, to_openeye

    @property
    def mda_mol(self):
        return self._mda_mol

    @mda_mol.setter
    def mda_mol(self, new_value):
        self._mda_mol = new_value

    @mda_mol.getter
    def mda_mol(self):
        if not self._mda_mol:
            import MDAnalysis as mda
            if self.file_path:
                self._mda_mol = mda.Universe(self.file_path)
            elif self.pdb_id:
                from ..databases.pdb import download_pdb_structure
                file_path = download_pdb_structure(self.pdb_id)
                self._mda_mol = mda.Universe(file_path)
            elif self._openeye_mol:
                from tempfile import NamedTemporaryFile
                from ..modeling.OEModeling import write_molecules

                with NamedTemporaryFile(suffix="pdb") as temp_file:
                    write_molecules([self._openeye_mol], temp_file.name)
                    self._mda_mol = mda.Universe(temp_file.name)
            else:
                raise ValueError(
                    "To allow access to MDAnalysis molecules, the `Protein`-like object needs to "
                    "be initialized with one of the following attributes:\npdb_id\nfile_path\n"
                    "mda_mol\nopeneye_mol"
                )

        return self._mda_mol

    @property
    def openeye_mol(self):
        return self._openeye_mol

    @openeye_mol.setter
    def openeye_mol(self, new_value):
        self._openeye_mol = new_value

    @openeye_mol.getter
    def openeye_mol(self):
        if not self._openeye_mol:
            from ..modeling.OEModeling import read_molecules
            if self.file_path:
                self._openeye_mol = read_molecules(self.file_path)[0]
            elif self.pdb_id:
                from ..databases.pdb import download_pdb_structure
                file_path = download_pdb_structure(self.pdb_id)
                self._openeye_mol = read_molecules(file_path)
            elif self._mda_mol:
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(suffix="pdb") as temp_file:
                    self._mda_mol.write(temp_file.name)
                    self._openeye_mol = read_molecules(temp_file.name)
            else:
                raise ValueError(
                    "To allow access to OpenEye molecules, the `Protein`-like object needs to "
                    "be initialized with one of the following attributes:\npdb_id\nfile_path\n"
                    "mda_mol\nopeneye_mol"
                )

        return self._openeye_mol


class KLIFSKinase(Protein):
    """
    Kinase object with access to KLIFS and supporting MDAnalysis and OpenEye toolkits.
    """

    def __init__(
            self,
            pdb_id="",
            file_path="",
            chain_id="",
            alternate_location="",
            ligand_name="",
            mda_mol=None,
            openeye_mol=None,
            name="",
            sequence="",
            uniprot_id=None,
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
            file_path=file_path,
            chain_id=chain_id,
            alternate_location=alternate_location,
            ligand_name=ligand_name,
            mda_mol=mda_mol,
            openeye_mol=openeye_mol,
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
            if self.structure_klifs_id and not self.kinase_klifs_id:
                structure_details = remote.structures.by_structure_klifs_id(
                    self.structure_klifs_id
                )
                self.kinase_klifs_id = structure_details["kinase.klifs_id"].iloc[0]

            if self.kinase_klifs_id:
                kinase_details = remote.kinases.by_kinase_klifs_id(self.kinase_klifs_id)
                self._kinase_klifs_sequence = kinase_details["kinase.pocket"].values[0]
            else:
                raise ValueError(
                    "To allow access to the kinase KLIFS sequence, the `Kinase` object needs to "
                    "be initialized with one of the following attributes:\nkinase_klifs_sequence"
                    "\nkinase_klifs_id\nstructure_klifs_id"
                )

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
