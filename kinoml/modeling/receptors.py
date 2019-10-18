"""
Modeling tools for macromolecular receptors, usually proteins
"""


def align_sequence(sequence, *targets):
    """
    Refactor this (only available as an executable; in the future we might
    want to have a `kinoml.utils.CommandlineWrapper` thingy)

    >>> def emboss_needle_search(emboss_needle, target_seq_path, template_seq_path):
    >>>     for template_seq in template_seq_path:
    >>>         target_seq_id = os.path.basename(target_seq_path).split('.')[0]
    >>>         template_seq_id = os.path.basename(template_seq).split('.')[0]
    >>>         print('{0} -sid1 {1} -asequence {2}/{3} -sid2 {4} -bsequence {5} -gapopen 10.0 -gapextend 0.5 -aformat3 markx3 -outfile {2}/protein_comp_modeling/protein_seq_alignment_files/{1}_{4}.needle'.format(emboss_needle, target_seq_id, os.getcwd(), target_seq_path, template_seq_id, template_seq))
    >>>         os.system('{0} -sid1 {1} -asequence {2}/{3} -sid2 {4} -bsequence {5} -gapopen 10.0 -gapextend 0.5 -aformat3 markx3 -outfile {2}/protein_comp_modeling/protein_seq_alignment_files/{1}_{4}.needle'.format(emboss_needle, target_seq_id, os.getcwd(), target_seq_path, template_seq_id, template_seq))

    """


def build_model_with_rosetta(query_sequence, alignment, structure_templates):
    """
    Refactor this to minimize file IO

    >>> templates = pd.read_csv(template_hits, sep=",")
    >>> target_seq = os.path.basename(target_seq_path).split('.')[0]
    >>> templates['tar_tem_seq_alin'] = templates['template'].apply(lambda x: "{}_{}.needle".format(target_seq, x))
    >>> templates['tar_tem_seq_alin'] = alignment_file_path+templates['tar_tem_seq_alin']
    >>> top_hit_template_file_path = templates['tar_tem_seq_alin'].tolist()
    >>>
    >>> aligned_seq = defaultdict(list)
    >>> for path in top_hit_template_file_path:
    >>>     target_template_file_name = os.path.splitext(os.path.basename(path))[0]
    >>>     target_name_fasta_format = '>{} ..'.format(target_template_file_name.split('_')[0])
    >>>     template_name_fasta_format = '>{} ..'.format('_'.join(target_template_file_name.split('_')[1:]))
    >>>     target_aligned_seq = ''
    >>>     template_aligned_seq = ''
    >>>     with open (path, 'r') as readFile:
    >>>         parse = False
    >>>         parse2 = False
    >>>         for line in readFile:
    >>>             line = line.strip()
    >>>             if not parse:
    >>>                 if line.startswith(target_name_fasta_format):
    >>>                     parse = True
    >>>             elif line.startswith(template_name_fasta_format):
    >>>                 parse = False
    >>>             else:
    >>>                 target_aligned_seq+=line
    >>>
    >>>             if not parse2:
    >>>                 if line.startswith(template_name_fasta_format):
    >>>                     parse2 = True
    >>>             elif line.startswith('#'):
    >>>                 parse2 = False
    >>>             else:
    >>>                 template_aligned_seq += line
    >>>     aligned_seq[target_template_file_name].append(target_aligned_seq)
    >>>     aligned_seq[target_template_file_name].append(template_aligned_seq)
    >>>
    >>> target_seq_for_modeling = {}
    >>> for name, alignment_file in aligned_seq.items():
    >>>     top_hits_alignment = '{}\n{}\n{}\n\n'.format(name, alignment_file[0], alignment_file[1])
    >>>     with open('protein_comp_modeling/top_hits_alignment.txt', 'a') as writeFile:
    >>>         writeFile.write(top_hits_alignment)
    >>>     target_seq_based_on_temp_pdb = ''
    >>>     for i in range(len(alignment_file[0])):
    >>>         if not alignment_file[1][i] == '-':
    >>>             target_seq_based_on_temp_pdb += alignment_file[0][i]
    >>>     target_seq_for_modeling[name]=target_seq_based_on_temp_pdb
    >>>
    >>> final_target_template_for_modeling = {}
    >>> for target_template, target_final_seq in target_seq_for_modeling.items():
    >>>     template_name = '_'.join(target_template.split('_')[1:])
    >>>     temp_list_dir = os.listdir(template_pdb_path)
    >>>     for template_hit in temp_list_dir:
    >>>         if template_name in template_hit:
    >>>             final_target_template_for_modeling[template_hit] = target_final_seq
    >>>
    >>> for template_pdb, target_seq in final_target_template_for_modeling.items():
    >>>     output_model_name = 'protein_comp_modeling/{}_{}.pdb'.format(target_seq_path.split('.')[0], '_'.join(template_pdb.split('_')[0:2]))
    >>>     join_apo_dir_path = os.path.join(template_pdb_path, template_pdb)
    >>>     pose = pyrosetta.pose_from_file(join_apo_dir_path)
    >>>     assert(pose.size() == len(target_seq))
    >>>     scorefxn = pyrosetta.get_fa_scorefxn()
    >>>     for i in range(len(target_seq)):
    >>>         seqpos = i + 1
    >>>         name1 = target_seq[i]
    >>>         if (name1 == "-"):
    >>>             continue
    >>>         pyrosetta.rosetta.protocols.toolbox.pose_manipulation.repack_this_residue(seqpos, pose, scorefxn, True, name1)
    >>>     pose.dump_pdb(output_model_name)
    """
