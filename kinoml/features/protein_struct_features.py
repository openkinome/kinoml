"""
protein.py
This is a tool to featurize kinase conformational changes through the entire Kinome.

"""

def key_klifs_residues(numbering):
    """
    Retrieve a list of PDB residue indices relevant to key kinase conformations mapped via KLIFS.

    Define indices of the residues relevant to a list of 12 collective variables relevant to
    kinase conformational changes. These variables include: angle between aC and aE helices,
    the key K-E salt bridge, DFG-Phe conformation (two distances), X-DFG-Phi, X-DFG-Psi,
    DFG-Asp-Phi, DFG-Asp-Psi, DFG-Phe-Phi, DFG-Phe-Psi, DFG-Phe-Chi1, and the FRET L-S distance.
    All features are under the current numbering of the structure provided.

    Parameters
    ----------
    numbering : list of int
        numbering[klifs_index] is the residue number for the given PDB file corresponding to KLIFS residue index 'klifs_index'

    Returns
    -------
    key_res : list of int
        Key residue indices

    """
    if numbering == None:
        print("The structure was not found in the klifs database.")
        key_res = None
        return key_res

    key_res = dict() #initialize key_res (which read from the 0-based numbering list)
    for i in range(5):
        key_res[f'group{i}'] = list()
    ## feature group 0: A-loop backbone dihedrals
    key_res['group0'].append(numbering[83]) # start of A-loop

    ## feature group 1: P-loop backbone dihedrals
    key_res['group1'].append(numbering[3]) # res0 in P-loop
    key_res['group1'].append(numbering[4]) # res1 in P-loop
    key_res['group1'].append(numbering[5]) # res2 in P-loop
    key_res['group1'].append(numbering[6]) # res3 in P-loop
    key_res['group1'].append(numbering[7]) # res4 in P-loop
    key_res['group1'].append(numbering[8]) # res5 in P-loop

    ## feature group 2: aC-related features
    #angle between aC and aE helices and the key salt bridge
    key_res['group2'].append(numbering[19])  # res0 in aC
    key_res['group2'].append(numbering[29])  # res10 in aC
    key_res['group2'].append(numbering[62])  # end of aE
    key_res['group2'].append(numbering[16])  # K in beta III
    key_res['group2'].append(numbering[23])  # E in aC

    ## feature group 3: DFG-related features
    key_res['group3'].append(numbering[79])  # X-DFG
    key_res['group3'].append(numbering[80])  # DFG-Asp
    key_res['group3'].append(numbering[81])  # DFG-Phe
    key_res['group3'].append(numbering[27])  # ExxxX

    ## feature group 4: the FRET distance
    # not in the list of 85 (equivalent to Aura"S284"), use the 100% conserved beta III K as a reference
    key_res['group4'].append(numbering[16] + 120)

    # not in the list of 85 (equivalent to Aura"L225"), use the 100% conserved beta III K as a reference
    key_res['group4'].append(numbering[16] + 61)

    return key_res

def compute_simple_protein_features(u, key_res):
    """
    This function takes the PDB code, chain id and certain coordinates of a kinase from
    a command line and returns its structural features.

    Parameters
    ----------
    u : object
	A MDAnalysis.core.universe.Universe object of the input structure (a pdb file or a simulation trajectory).
    key_res : dict of int
        A dictionary (with keys 'group0' ... 'group4') of feature-related residue indices in five feature groups.
    Returns
    -------
    features: list of floats
    	A list (single structure) or lists (multiple frames in a trajectory) of 72 features in 5 groups (A-loop, P-loop, aC, DFG, FRET)

    .. todo :: Use kwargs with sensible defaults instead of relying only on positional arguments.


    """
    from MDAnalysis.core.groups import AtomGroup
    from MDAnalysis.analysis.dihedrals import Dihedral
    from MDAnalysis.analysis.distances import dist
    import numpy as np
    import pandas as pd

    # get the array of atom indices for the calculation of:
    #       * seven dihedrals (a 7*4 array where each row contains indices of the four atoms for each dihedral)
    #       * two ditances (a 2*2 array where each row contains indices of the two atoms for each dihedral)
    dih = np.zeros(shape=(7, 4), dtype=int, order='C')
    dis = np.zeros(shape=(2, 2), dtype=int, order='C')

    # name list of the dihedrals and distances
    dih_names = ['xDFG_phi', 'xDFG_psi', 'dFG_phi', 'dFG_psi', 'DfG_phi',
        'DfG_psi', 'DfG_chi1'
    ]
    dis_names = ['DFG_conf1', 'DFG_conf2', 'DFG_conf3', 'DFG_conf4']

    # parse the topology info (0-based atom indices)

    ### dihedrals (feature group 3)
    # dihedral 0 & 1: X-DFG Phi & Psi
    dih[0][0] = int(u.select_atoms(f"resid {key_res['group3'][0]-1} and name C")[0].ix) # xxDFG C
    dih[0][1] = int(u.select_atoms(f"resid {key_res['group3'][0]} and name N")[0].ix) # xDFG N
    dih[0][2] = int(u.select_atoms(f"resid {key_res['group3'][0]} and name CA")[0].ix) # xDFG CA
    dih[0][3] = int(u.select_atoms(f"resid {key_res['group3'][0]} and name C")[0].ix) # xDFG C
    dih[1][0] = dih[0][1] # xDFG N
    dih[1][1] = dih[0][2] # xDFG CA
    dih[1][2] = dih[0][3] # xDFG C
    dih[1][3] = int(u.select_atoms(f"resid {key_res['group3'][1]} and name N")[0].ix) # DFG-Asp N

    # dihedral 2 & 3: DFG-Asp Phi & Psi
    dih[2][0] = dih[0][3] # xDFG C
    dih[2][1] = dih[1][3] # DFG-Asp N
    dih[2][2] = int(u.select_atoms(f"resid {key_res['group3'][1]} and name CA")[0].ix) # DFG-Asp CA
    dih[2][3] = int(u.select_atoms(f"resid {key_res['group3'][1]} and name C")[0].ix) # DFG-Asp C
    dih[3][0] = dih[2][1] # DFG-Asp N
    dih[3][1] = dih[2][2] # DFG-Asp CA
    dih[3][2] = dih[2][3] # DFG-Asp C
    dih[3][3] = int(u.select_atoms(f"resid {key_res['group3'][2]} and name N")[0].ix) # DFG-Phe N

    # dihedral 4 & 5: DFG-Phe Phi & Psi
    dih[4][0] = dih[2][3] # DFG-Asp C
    dih[4][1] = dih[3][3] # DFG-Phe N
    dih[4][2] = int(u.select_atoms(f"resid {key_res['group3'][2]} and name CA")[0].ix) # DFG-Phe CA
    dih[4][3] = int(u.select_atoms(f"resid {key_res['group3'][2]} and name C")[0].ix) # DFG-Phe C
    dih[5][0] = dih[4][1] # DFG-Phe N
    dih[5][1] = dih[4][2] # DFG-Phe CA
    dih[5][2] = dih[4][3] # DFG-Phe C
    dih[5][3] = int(u.select_atoms(f"resid {key_res['group3'][2]+1} and name N")[0].ix) # DFG-Gly N

    # dihedral 6: DFG-Phe Chi1
    dih[6][0] = dih[3][3] #DFG-Phe N
    dih[6][1] = dih[4][2] #DFG-Phe CA
    dih[6][2] = int(u.select_atoms(f"resid {key_res['group3'][2]} and name CB")[0].ix) # DFG-Phe CB
    dih[6][3] = int(u.select_atoms(f"resid {key_res['group3'][2]} and name CG")[0].ix) # DFG-Phe CG

    ### distances
    ## Dunbrack distances D1, D2
    dis[0][0] = int(u.select_atoms(f"resid {key_res['group3'][3]} and name CA")[0].ix) # ExxxX CA
    dis[0][1] = int(u.select_atoms(f"resid {key_res['group3'][2]} and name CZ")[0].ix) # DFG-Phe CZ
    dis[1][0] = int(u.select_atoms(f"resid {key_res['group2'][3]} and name CA")[0].ix) # K in beta III CA 
    dis[1][1] = dis[0][1] # DFG-Phe CZ

    # check if there is any missing coordinates; if so, skip dihedral/distance calculation for those residues
    check_flag = 1
    for i in range(len(dih)):
        if 0 in dih[i]:
            dih[i] = [0,0,0,0]
            check_flag = 0

    for i in range(len(dis)):
        if 0 in dis[i]:
            dis[i] = [0,0]
            check_flag = 0
    if check_flag:
        print("There is no missing coordinates.  All dihedrals and distances will be computed.")

    # compute dihedrals and distances
    distances = list()
    dih_ags = list()
    for i in range(7): # for each of the dihedrals
        dih_ags.append(AtomGroup(dih[i], u))
    dihedrals = Dihedral(dih_ags).run().angles

    each_frame = list()
    for i in range(2):
        ag0 = AtomGroup([dis[i][0]], u) # first atom in each atom pair
        ag1 = AtomGroup([dis[i][1]], u) # second atom in each atom pair
        each_frame.append(dist(ag0, ag1)[-1][0])
    each_frame = np.array(each_frame)
    distances.append(each_frame)
    # clean up
    del u, dih, dis
    return dihedrals, distances
