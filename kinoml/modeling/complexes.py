def create_protein_ligand_complex(protein, ligand):
    """
    Refactor this function to minimize IO

    >>> def protein_ligand_concatenation (template_hits, target_fasta_file, ligands):
    >>>     templates_df = pd.read_csv(template_hits, sep=",")
    >>>     # first templatePDB in a template Column
    >>>     top_first_hit_pdb = templates_df["template"].iloc[0]
    >>>     target_pdb_path = os.path.dirname(template_hits)
    >>>     target_pdb_name = os.path.basename(target_fasta_file).split('.')[0]
    >>>     top_hit_model = '{0}_{1}.pdb'.format(target_pdb_name, top_first_hit_pdb)
    >>>     for lig in ligands:
    >>>         protein_model = os.path.join(target_pdb_path, top_hit_model)
    >>>         basename_ligand = os.path.splitext(os.path.basename(lig))[0]
    >>>         complex_protein_ligand = '{0}_{1}.pdb'.format(top_hit_model.split('.')[0], basename_ligand)
    >>>         print('cat {0} {1} > protein_ligand_complex_top_1_comp_model/{2}'.format(protein_model, lig, complex_protein_ligand))
    >>>         os.system('cat {0} {1} > protein_ligand_complex_top_1_comp_model/{2}'.format(protein_model, lig, complex_protein_ligand))
    """