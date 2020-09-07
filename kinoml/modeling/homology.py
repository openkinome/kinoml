from ..core.proteins import ProteinStructure
from ..core.sequences import KinaseDomainAminoAcidSequence


class HomologyModel:  #  TODO inherent a Base class?


    def __init__(self, *args, **kwargs):
        from appdirs import user_cache_dir

        self.alipath = f"{user_cache_dir()}/alignment.ali"
        #  TODO specify id, template, and sequence here? e.g.:

        #  self.identifier = identifier
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
            if backend == "uniprot":
                sequence = KinaseDomainAminoAcidSequence.from_uniprot(identifier)
            elif backend == "ncbi":
                sequence = KinaseDomainAminoAcidSequence.from_ncbi(identifier)
            else:
                raise ValueError(
                    'Backend "%s" not supported. Please choose from ["uniprot", "ncbi"]'
                    % (backend)
                )

        else:
            params = {"query": identifier, "format": "fasta"}
            response = requests.get("http://www.uniprot.org/uniprot/", params)

            sequence = response.text.split("\n", 1)[1].replace("\n", "")

        return sequence


    def get_alignment(
        self, template_system, canonical_sequence, pdb_entry=False, window=15
    ):
        #  WIP
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


    def get_model(
        self, template_system: ProteinStructure, alignment, num_models: int = 100
    ):
        """
        Generate homology model(s) based on a template and alignment with MODELLER
        Parameters
        ----------
        template_system: ProteinStructure
            A ProteinStructure object to use as the protein template
        alignment: str
            Path to alignment file generated by HomologyModel.get_alignment()
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
