from ..core.proteins import ProteinStructure
from ..core.sequences import KinaseDomainAminoAcidSequence
from .alignment import Alignment

class HomologyModel:  #  TODO inherent a Base class?
    def __init__(self, metadata=None, *args, **kwargs):
        #  TODO specify id, template, and sequence here? e.g.:

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        # self.identifier = identifier
        #  self.template = template
        #  self.sequence = sequence

    def get_pdb_template(
        self,
        sequence,
    ):
        """
        Retrieve a template structure from PDB from a BLAST search
        Parameters
        ----------
        sequence: str
            A string of the protein sequence

        Returns
        -------
        hits: dict
            A dictionary generated from ProDy with PDB hits.
        """

        from prody import blastPDB
        import tempfile
        import pickle

        blast_record = blastPDB(sequence)
        hits = blast_record.getHits()

        #  TODO need to address when BLAST search times out
        #  TODO add option based on sequency similarity cut off

        return hits

    def get_sequence(
        self, identifier: str, kinase: bool = False, backend: str = "uniprot"
    ):
        """
        Retrieve a sequence based on an identifier code
        Parameters
        ----------
        identifier: str
            A string of the identifier to query (e.g 'P04629' from UniProt)
        kinase: bool
            Specify whether the sequence search should query kinase domains
        backend: str
            Specify which database to query. Options = ["uniprot", "ncbi"]

        Returns
        -------
        sequence: str
            A protein sequence.
        """

        import requests
        from io import StringIO

        if kinase:
            try:
                from_method = getattr(KinaseDomainAminoAcidSequence, f"from_{backend}")
            except AttributeError:
                raise ValueError(
                    'Backend "%s" not supported. Please choose from ["uniprot", "ncbi"]'
                    % (backend)
                )
            else:
                sequence = from_method(identifier)

        else:
            params = {"query": identifier, "format": "fasta"}
            response = requests.get("http://www.uniprot.org/uniprot/", params)
            sequence = response.text.split("\n", 1)[1].replace("\n", "")

        return sequence

    def make_model(
        self,
        template_system: ProteinStructure,
        target_sequence: KinaseDomainAminoAcidSequence,
        alignment: Alignment,
        num_models: int = 100,
    ):
        """
        Generate homology model(s) based on a template and alignment with MODELLER
        Parameters
        ----------
        template_system: ProteinStructure
            The template system.
        target_sequence: KinaseDomainAminoAcidSequence
            The target sequence
        alignment: Alignment
            An Alignment object containing information on aligned sequences. These
            sequences must be the same as present in template_system and target_sequence
        num_models: int
            The number of homology models to generate. default = 100
        Returns
        -------
        """

        from modeller import log, environ
        from modeller.automodel import dope_loopmodel, assess

        log.verbose()
        env = environ()

        pdb_id = template_system.metadata["id"]

        env.io.atom_files_directory = [
            template_system.metadata["path"].split(".")[0].split(pdb_id)[0]
        ]

        # TODO handle usage of "ncbi" instead of "uniprot_id"
        a = dope_loopmodel(
            env,
            alnfile=alignment.alignment_file_path,
            knowns=template_system.metadata["id"],
            sequence=target_sequence.metadata["uniprot_id"],
            loop_assess_methods=(assess.DOPE, assess.GA341),
        )

        a.starting_model = 1
        a.ending_model = num_models

        # TODO could add loop refinement option here
        # TODO need to specify output directory for models
        a.make()

