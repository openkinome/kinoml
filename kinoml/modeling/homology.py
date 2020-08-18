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

        # add the sequences
        aln.append_sequence(sequence_1, blank_single_chain=True)
        aln.append_sequence(sequence_2, blank_single_chain=True)

        aln[0].code = 'seq1'
        aln[1].code = 'seq2'

        # aln[0].atom_file = "4yne.pdb"
        # aln[0].prottyp = 'structureX'

        # align the sequences
        aln.align()

        with tempfile.NamedTemporaryFile(suffix=".ali") as temp_file:
            aln.write(file=temp_file)
            temp_file.seek(0) # rewind the file for reading

            ali_lines = []
            for line in temp_file.readlines():
                line_str = line.decode("utf-8")
                ali_lines.append(line_str.strip())

            index = ali_lines.index(">P1;seq2")

            ali_1 = ali_lines[:index][1:-1] # template
            ali_2 = ali_lines[index:] # target

            ali_1_new, ali_2_new = [], []

            # need to run through each list with 
            for i, (a, b) in enumerate(zip(ali_1, ali_2)):

                if any(c.isalpha() for c in a): 
                    ali_1_new.append(ali_1[i])
                    ali_2_new.append(ali_2[i])

            # # DEBUGGING
            # for i, line in enumerate(temp_file.readlines()):
            #     print(line.decode("utf-8"))

        raise NotImplementedError
    

    def get_model():
        raise NotImplementedError

