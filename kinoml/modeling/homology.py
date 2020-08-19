from ..core.proteins import ProteinStructure


class HomologyModel:  #  TODO inherent a Base class?

    """
    Given a UniProt identifier, generate a template on to which 
    a homology model can be made and constrcut a homology model.

    The passed sequence will be used to generate a PDB template 
    structure (based on a BLAST search) or, if provided, use a 
    pre-generated template. This will be used to produce a final 
    homology model.
    """

    def __init__(self, name="", *args, **kwargs):
        from appdirs import user_cache_dir

        self.name = name
        self.path = f"{user_cache_dir()}/{self.name}.ali"
        #  TODO specify id, template, and sequence here? e.g.:

        #  self.identifier = identifier
        #  self.template = template
        #  self.sequence = sequence


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

        with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
            pickle.dump(blast_record, temp_file)

            blast_record = pickle.load(open(temp_file.name, "rb"))
            best = blast_record.getBest()["pdb_id"]

            #  TODO add option based on sequency similarity cut off

            top_pdb_template = ProteinStructure.from_name(best)

        return top_pdb_template

    def get_uniprot_sequence(self, identifier: str):
        import requests
        from io import StringIO

        params = {"query": identifier, "format": "fasta"}
        response = requests.get("http://www.uniprot.org/uniprot/", params)

        up_sequence = response.text.split("\n", 1)[1].replace("\n", "")

        #  TODO add option to specify just the kinase domain

        return up_sequence

    def get_alignment(self, template_system, canonical_sequence):

        #  TODO write output to a logger
        import tempfile
        from modeller import alignment, log, environ

        log.verbose()
        env = environ()

        aln = alignment(env)

        # add the sequences
        aln.append_sequence(template_system.sequence.sequence, blank_single_chain=True)
        aln.append_sequence(canonical_sequence, blank_single_chain=True)

        aln[0].code = template_system.metadata['id']
        aln[1].code = "target_seq" #  TODO set target sequence name to be UniProt ID

        # align the sequences
        aln.align()

        with tempfile.NamedTemporaryFile(suffix=".ali") as temp_file:
            aln.write(file=temp_file)
            temp_file.seek(0)  # rewind the file for reading

            ali_lines = []
            for line in temp_file.readlines():
                line_str = line.decode("utf-8")
                ali_lines.append(line_str.strip())

            # split the list for easy reading
            index = ali_lines.index(">P1;target_seq")
            ali_1 = ali_lines[:index][1:-1]  # template
            ali_2 = ali_lines[index:]  # target

            ali_1_new, ali_2_new = [], []

            # remove long blank regions in template seq "-"
            for i, (a, b) in enumerate(zip(ali_1, ali_2)):

                if any(c.isalpha() for c in a):
                    ali_1_new.append(ali_1[i])
                    ali_2_new.append(ali_2[i])  # remove corresponding seq in target

            #  TODO change sequence numbers in alignment file

        # write new alignment file without long blank regions
        with open(self.path, 'w') as ali_file: # saving the file to cache
            for item in ali_1_new:
                ali_file.write("%s\n" % item)
            for item in ali_2_new:
                ali_file.write("%s\n" % item)
        

    def get_model(self, template_structure, alignment):
        raise NotImplementedError
