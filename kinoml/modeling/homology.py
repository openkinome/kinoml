from ..core.proteins import ProteinStructure
from typing import Union


class HomologyModel(): # inherent a Base class? 

    """
    Given a UniProt identifier, generate a template on to which 
    a homology model can be made and constrcut a homology model.

    The passed sequence will be used to generate a PDB template 
    structure (based on a BLAST search) or, if provided, use a 
    pre-generated template. This will be used to produce a final 
    homology model.
    """

    def __init__(self, identifier=None, template=None, sequence=None, *args, **kwargs):
        self. identifier = identifier
        self.template = template
        self.sequence = sequence


    def get_pdb_template(self, sequence):
        """
        Retrieve a template structure from PDB from a BLAST search
        Parameters
        ----------
        sequence: str
            A string of the protein sequence
        Returns
        -------
        pdb_template: ProteinStructure
            A ProteinStructure object generated from retrieval from a PDB BLAST search.
        """
        
        from prody import blastPDB
        import tempfile
        import pickle

        blast_record = blastPDB(sequence)

        with tempfile.NamedTemporaryFile(suffix='.pkl') as temp_file:
            pickle.dump(blast_record, temp_file)

            blast_record = pickle.load(open(temp_file.name, 'rb'))
            best = blast_record.getBest()['pdb_id']

            top_pdb_template = ProteinStructure.from_name(best)

        return top_pdb_template

    def get_uniprot_sequence(self, identifier: str):
        import requests
        from io import StringIO

        params = {"query": identifier, "format": "fasta"}
        response = requests.get("http://www.uniprot.org/uniprot/", params)

        up_sequence = response.text.split('\n', 1)[1].replace('\n','')

        return up_sequence


    def get_alignment(self, sequence_1, sequence_2):

        import tempfile
        from modeller import alignment, log, environ
        log.verbose()
        env = environ()

        aln = alignment(env)
        aln.append_sequence(sequence_1, blank_single_chain=True)
        aln.append_sequence(sequence_2, blank_single_chain=True)

        aln[0].code = 'seq1'
        aln[1].code = 'seq2'

        aln.edit(edit_align_codes='seq1', base_align_codes='seq2',
         min_base_entries=1, overhang=3)

        aln.align()

        with tempfile.NamedTemporaryFile(suffix=".ali") as temp_file:
            aln.write(file=temp_file)
            temp_file.seek(0) # rewind the file for reading
        
        #TODO: edit alignment to remove long indels 

            # # DEBUGGING
            # for i, line in enumerate(temp_file.readlines()):
            #     print(line.decode("utf-8"))

        return alignment_file
    

    def get_model():
        raise NotImplementedError

