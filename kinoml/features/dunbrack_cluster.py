'''
Tools to assign a structure or a trajectory of structures into 
conformational clusters based on Modi and Dunbrack, 2019 (https://pubmed.ncbi.nlm.nih.gov/30867294/) 
'''
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
