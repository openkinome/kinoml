'''
Tools to assign a structure or a trajectory of structures into 
conformational clusters based on Modi and Dunbrack, 2019 (https://pubmed.ncbi.nlm.nih.gov/30867294/) 
'''
from pathlib import Path
import tempfile

from appdirs import user_cache_dir
import pandas as pd


def assign(dihedrals, distances):
    from math import cos
    import numpy as np
    # define the centroid values for Dunbrack features
    centroid = dict()
    centroid[(0, 'x_phi')]=  -129.0
    centroid[(0, 'x_psi')]=   179.0
    centroid[(0, 'd_phi')]=    61.0
    centroid[(0, 'd_psi')]=    81.0
    centroid[(0, 'f_phi')]=   -97.0
    centroid[(0, 'f_psi')]=    20.0
    centroid[(0, 'f_chi1')]=  -71.0

    centroid[(1, 'x_phi')]=  -119.0
    centroid[(1, 'x_psi')]=   168.0
    centroid[(1, 'd_phi')]=    59.0
    centroid[(1, 'd_psi')]=    34.0
    centroid[(1, 'f_phi')]=   -89.0
    centroid[(1, 'f_psi')]=    -8.0
    centroid[(1, 'f_chi1')]=   56.0

    centroid[(2, 'x_phi')]=  -112.0
    centroid[(2, 'x_psi')]=    -8.0
    centroid[(2, 'd_phi')]=  -141.0
    centroid[(2, 'd_psi')]=   148.0
    centroid[(2, 'f_phi')]=  -128.0
    centroid[(2, 'f_psi')]=    23.0
    centroid[(2, 'f_chi1')]=  -64.0

    centroid[(3, 'x_phi')]=  -135.0
    centroid[(3, 'x_psi')]=   175.0
    centroid[(3, 'd_phi')]=    60.0
    centroid[(3, 'd_psi')]=    65.0
    centroid[(3, 'f_phi')]=   -79.0
    centroid[(3, 'f_psi')]=   145.0
    centroid[(3, 'f_chi1')]=  -73.0

    centroid[(4, 'x_phi')]=  -125.0
    centroid[(4, 'x_psi')]=   172.0
    centroid[(4, 'd_phi')]=    60.0
    centroid[(4, 'd_psi')]=    33.0
    centroid[(4, 'f_phi')]=   -85.0
    centroid[(4, 'f_psi')]=   145.0
    centroid[(4, 'f_chi1')]=   49.0

    centroid[(5, 'x_phi')]=  -106.0
    centroid[(5, 'x_psi')]=   157.0
    centroid[(5, 'd_phi')]=    69.0
    centroid[(5, 'd_psi')]=    21.0
    centroid[(5, 'f_phi')]=   -62.0
    centroid[(5, 'f_psi')]=   134.0
    centroid[(5, 'f_chi1')]= -145.0

    assignment = list()
    for i in range(len(distances)):
        ## reproduce the Dunbrack clustering
        ## level1: define the DFG positions
        if distances[i][0] <= 11.0 and distances[i][1] <= 11.0: # angstroms
            ## can only be BABtrans
            assignment.append(7)
        elif distances[i][0] > 11.0 and distances[i][1] < 14.0:
            ## can only be BBAminus
            assignment.append(6)
        else:
            ## belong to DFGin and possibly clusters 0 - 5
            mindist=10000.0
            cluster_assign = 0
 
            for c in range(6):
                total_dist = float((2.0 * (1.0-cos((dihedrals[i][0] - centroid[(c, 'x_phi')])*np.pi / 180.0)))
                + (2.0 * (1.0-cos((dihedrals[i][1] - centroid[(c, 'x_psi')])*np.pi / 180.0)))
                + (2.0 * (1.0-cos((dihedrals[i][2] - centroid[(c, 'd_phi')])*np.pi / 180.0)))
                + (2.0 * (1.0-cos((dihedrals[i][3] - centroid[(c, 'd_psi')])*np.pi / 180.0)))
                + (2.0 * (1.0-cos((dihedrals[i][4] - centroid[(c, 'f_phi')])*np.pi / 180.0)))
                + (2.0 * (1.0-cos((dihedrals[i][5] - centroid[(c, 'f_psi')])*np.pi / 180.0)))
                + (2.0 * (1.0-cos((dihedrals[i][6] - centroid[(c, 'f_chi1')])*np.pi / 180.0)))) / 7
                if total_dist < mindist:
                    mindist = total_dist
                    clust_assign = c
            assignment.append(clust_assign)
    return assignment 


class PDBDunbrack:

    _PDB_DUNBRACK_LIBRARY = Path(f"{user_cache_dir()}/pdb_dunbrack_library.csv")

    def __init__(self):
        self.pdb_dunbrack_library = self.update()

    def __repr__(self):
        return f"<{self._PDB_DUNBRACK_LIBRARY}>"

    def update(self, reinitialize: bool = False) -> pd.DataFrame:
        """
        Update DataFrame holding information about kinases from the KLIFS database and the corresponding Dunbrack
        cluster.
        Parameters
        ----------
        reinitialize: bool
            If the DataFrame should be built from scratch.
        Returns
        -------
        pdb_dunbrack_library: pd.DataFrame
            DataFrame holding information about kinases from KLIFS and the corresponding Dunbrack cluster.
        """
        from .klifs import query_klifs_database
        import klifs_utils
        import MDAnalysis as mda
        from .protein_struct_features import key_klifs_residues, compute_simple_protein_features
        from tqdm import tqdm

        # get available kinase information from KLIFS
        klifs_kinase_ids = klifs_utils.remote.kinases.kinase_names().kinase_ID.to_list()
        klifs_kinase_df = klifs_utils.remote.structures.structures_from_kinase_ids(klifs_kinase_ids)

        # initialize library
        if not self._PDB_DUNBRACK_LIBRARY.is_file() or reinitialize is True:
            columns = list(klifs_kinase_df.columns) + ["dunbrack_cluster"]
            pdb_dunbrack_library = pd.DataFrame(columns=columns)
            pdb_dunbrack_library.to_csv(self._PDB_DUNBRACK_LIBRARY, index=False)

        pdb_dunbrack_library = pd.read_csv(self._PDB_DUNBRACK_LIBRARY)

        counter = 0
        for index, row in tqdm(klifs_kinase_df.iterrows(), total=klifs_kinase_df.shape[0]):
            structure_id = row["structure_ID"]
            if structure_id not in list(pdb_dunbrack_library["structure_ID"]):
                counter += 1
                try:  # assign dunbrack cluster
                    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+t") as temp_file:
                        pdb_text = klifs_utils.remote.coordinates.complex._complex_pdb_text(structure_id)
                        temp_file.write(pdb_text)
                        u = mda.Universe(temp_file.name)
                        klifs = query_klifs_database(row["pdb"], row["chain"])
                        key_res = key_klifs_residues(klifs['numbering'])
                        dihedrals, distances = compute_simple_protein_features(u, key_res)
                        assignment = assign(dihedrals, distances)[0]
                except:  # catch all errors and assign None
                    assignment = None
                row["dunbrack_cluster"] = assignment
                pdb_dunbrack_library = pdb_dunbrack_library.append(row, ignore_index=True)
                if counter % 10 == 0:  # save every 10th structure, so one can pause in between
                    pdb_dunbrack_library.to_csv(self._PDB_DUNBRACK_LIBRARY, index=False)

        pdb_dunbrack_library.to_csv(self._PDB_DUNBRACK_LIBRARY, index=False)
        return pdb_dunbrack_library

    def structures_by_cluster(self, cluster_id):
        """

        """
        import pandas as pd
        pdb_dunbrack_library = pd.read_csv(self._PDB_DUNBRACK_LIBRARY)
        structures = pdb_dunbrack_library[pdb_dunbrack_library["dunbrack_cluster"] == cluster_id]
        return structures
