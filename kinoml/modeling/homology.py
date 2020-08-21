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

    def __init__(self, *args, **kwargs):
        from appdirs import user_cache_dir

        self.alipath = f"{user_cache_dir()}/alignment.ali"
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
        best = blast_record.getBest()["pdb_id"]

        top_pdb_template = ProteinStructure.from_name(best)

        #  TODO add option based on sequency similarity cut off
        #  TODO add option to return all pdb models, not just the best

        return top_pdb_template

    def get_uniprot_sequence(self, identifier: str):
        import requests
        from io import StringIO

        params = {"query": identifier, "format": "fasta"}
        response = requests.get("http://www.uniprot.org/uniprot/", params)

        up_sequence = response.text.split("\n", 1)[1].replace("\n", "")

        #  TODO add option to specify just the kinase domain

        return up_sequence

    def get_alignment(
        self, template_system, canonical_sequence, pdb_entry=False, window=15
    ):

        #  TODO write output to a logger
        import tempfile
        import requests
        from modeller import alignment, log, environ, model
        import numpy as np
        import re

        log.verbose()
        env = environ()

        pdb_id = template_system.metadata["id"]

        if pdb_entry == True:
            from appdirs import user_cache_dir

            template_system.metadata["path"] = f"{user_cache_dir()}/{pdb_id}.pdb"

            # TODO there is probably a better way to this, it's a
            # repeat of ..core.proteins.ProteinStructure.from_name
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url)

            with open(
                template_system.metadata["path"], "wb"
            ) as pdb_file:  # saving the pdb to cache
                pdb_file.write(response.content)

        env.io.atom_files_directory = [
            template_system.metadata["path"].split(".")[0].split(pdb_id)[0]
        ]

        aln = alignment(env)
        mdl = model(env)

        # Read the whole template structure file, gets sequence automatically
        code = template_system.metadata["id"]
        mdl.read(file=code, model_segment=("FIRST:@", "END:"))

        # Add the template sequence to the alignment
        aln.append_model(mdl, align_codes=code, atom_files=code)

        # add the canonical target sequence
        aln.append_sequence(canonical_sequence, blank_single_chain=True)

        # edit canonical target sequence keywords
        aln[1].code = "target_seq"  #  TODO set target sequence name to be UniProt ID?

        # align the sequences TODO should we use salign() instead?
        aln.align()

        with tempfile.NamedTemporaryFile(suffix=".ali") as temp_file:
            aln.write(file=temp_file)
            temp_file.seek(0)  # rewind the file for reading
            # TODO add option to save original alignment

            ali_lines = []
            for line in temp_file.readlines():
                line_str = line.decode("utf-8")
                ali_lines.append(line_str.strip())

            # split the list for easy reading
            index = ali_lines.index(">P1;target_seq")
            ali_1 = ali_lines[:index][1:-1]  # template
            ali_2 = ali_lines[index:]  # target

            ali_1_new, ali_2_new = [], []
            ali_1_final, ali_2_final = [], []

            # remove long blank regions in template seq "-"
            for i, (a, b) in enumerate(zip(ali_1, ali_2)):

                if any(line_element.isalpha() for line_element in a):
                    ali_1_new.append(ali_1[i])
                    ali_2_new.append(ali_2[i])  # remove corresponding seq in target

            #  TODO change sequence numbers in alignment file,
            # e.g. aln[0].range = ['3:', ':100']

            for i, (line_ali_1, line_ali_2) in enumerate(zip(ali_1_new, ali_2_new)):
                if i > 1:  # ignore the title lines, only focus on sequence

                    index_dict = dict.fromkeys(np.arange(len(line_ali_1)).tolist())

                    for i in range(len(line_ali_1) - window + 1):
                        gap_count = 0
                        chunk1 = line_ali_1[i : i + window]
                        chunk2 = line_ali_1[i + window : i + (2 * window)]

                        # check if there are '-'s in the chunk
                        if not any(c.isalpha() for c in chunk1):
                            for c1, c2 in zip(chunk1, chunk2):

                                # check each character in chunk1 & chunk2
                                if re.match("-", c1) and re.match("-", c2) is not None:
                                    gap_count += 1

                            # if we have a large region of '-'s, mark for deletion
                            if gap_count == window:
                                for key in np.arange(i, i + (2 * window)):
                                    index_dict[key] = "delete"

                    new_template_string = []
                    for i, s in enumerate(line_ali_1):
                        if index_dict[i] is None:
                            new_template_string.append(line_ali_1[i])

                    ali_1_final.append("".join(new_template_string))

                    new_target_string = []
                    for i, s in enumerate(line_ali_1):
                        if index_dict[i] is None:
                            new_target_string.append(line_ali_2[i])

                    ali_2_final.append("".join(new_target_string))

                else:
                    continue

        with open(self.alipath, "w") as ali_file:  # saving the file to cache

            for i, item in enumerate(ali_1_new):
                if i < 2:
                    ali_file.write("%s\n" % item)  # write the template title lines
                else:
                    continue
            for item in ali_1_final:
                ali_file.write("%s\n" % item)  # write the template sequence lines

            for i, item in enumerate(ali_2_new):
                if i < 2:
                    ali_file.write("%s\n" % item)  # write the target title lines
                else:
                    continue
            for item in ali_2_final:
                ali_file.write("%s\n" % item)  # write the target sequence lines


    def get_model(self, template_system, alignment, num_models):
        from modeller import log, environ
        from modeller.automodel import dope_loopmodel, assess

        log.verbose()
        env = environ()

        pdb_id = template_system.metadata["id"]

        env.io.atom_files_directory = [
            template_system.metadata["path"].split(".")[0].split(pdb_id)[0]
        ]

        a = dope_loopmodel(
            env,
            alnfile=alignment,
            knowns=template_system.metadata["id"],
            sequence="target_seq",
            loop_assess_methods=(assess.DOPE, assess.GA341),
        )

        a.starting_model = 1
        a.ending_model = num_models

        # TODO could add loop refinement option here
        # TODO files are written out to ./, need to move after generation
        # no way to specify output dir according to modeller docs

        a.make()
