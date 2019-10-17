"""
Workflows for kinase modeling, based on Karanicolas' protocols

Given a ligand and a target kinase, this workflow will:

Part A: preparation

    1. Align the ligand against an in-house database of active kinase ligands
       (resulting in a total of 100 conformers).
    2. Align the kinase sequence against an in-house database of kinase sequences

Part B1: build best models

    3. Use Rosetta to build 10 most promising models, as suggested by the alignment
       similarity.
    4. Best model will be combined with conformers described in step 1 (100 total
       complexes)
    5. Minimize them.

Part B2: build remaining models

    6. Pick 10 best ligands from step 1, and all 10 models from step 10. Build the
       resulting 100 models.
    7. Minimize them.

Part C: Results

    8. Report results from B1 and B2


Notes
-----

Originally developed at https://github.com/karanicolaslab/kinmodel.

"""

import os
import numpy as np
from openforcefield.topology import Molecule


def parse_cli():
    p = ArgumentParser()
    p.add_argument("ligand")
    p.add_argument("protein")
    p.add_argument("--nconformers", default=100, help="Number of ligand conformers that will be generated")
    p.add_argument("--nmodels", default=10, help="Number of target models that will be built")
    p.add_argument("--ligand_library", default=None)
    p.add_argument("--sequence_library", default=None)

    return p.parse_args()


def _load_ligand(ligand):
    """
    Load a ligand using OpenForceField toolkit.

    Parameters
    ----------
    ligand : str
        This can be a path to a file (supported by OFF) or a SMILES string

    Returns
    -------
    openforcefield.topology.Molecule
    """
    if os.path.isfile(ligand):
        return Molecule.from_file(ligand)
    return Molecule.from_smiles(ligand)


def _superpose_ligand(ligand, dataset):
    """
    Structural superposition of a query molecule ``ligand`` against an existing
    dataset of small compounds

    Parameters
    ----------
    ligand : openforcefield.topology.Molecule
        Query ligand that will be superposed. Conformers must have been generated
        previously.
    dataset : str
        Path to a SDF file containing the target database

    Returns
    -------
    superposed_conformers : array, shape=len(dataset)*len(ligand.conformers)*ligand.n_atoms*3
    scores : array, shape=len(dataset)*len(ligand.conformers)
    """
    pass


def _align_sequence():
    pass


def main():
    # Temporarily add imports here to have a broader picture of the code

    args = parse_cli()

    # Part A1 - align ligands
    ligand_molecule = _load_ligand(args.ligand)
    ligand_molecule.generate_conformers(n_conformers=args.nconformers)

    superposed_xyz, superposition_scores = _superpose_ligand(ligand_molecule, args.ligand_library)

    # TODO: Get best conformers: see https://stackoverflow.com/a/38884051
    best_conformers_indices = np.argpartition(superposition_scores, -args.nconformers)[-args.nconformers:]
    best_conformers = superposed_xyz[best_conformers_indices]  # FIXME: This does not work as expected

    # Part A2 - align protein sequence and build protein models
    protein_molecule = _load_protein(args.protein)
    protein_sequence = protein_molecule.sequence
    aligned_sequences, alignment_scores = _align_sequence(protein_sequence, args.sequence_library)
    best_alignment_indices = np.argpartition(alignment_scores, -args.nmodels)[-args.nmodels:]
    candidate_sequences = aligned_sequences[best_alignment_indices]  # FIXME
    built_models = build_with_rosetta(template, sequences, n_results=10)  # TODO: parallel call!

    # Part B1 - build protein-ligand complexes and minimize
    complexes = []
    for protein, ligand in product(built_models, best_conformers):
        complexes.append(concatenate(protein, ligand))

    # TODO: parallel call
    minimized_complexes = [minimize(c) for c in complexes]

    # Part B2 - repeat for remaining models

    # Part C - report
    report(...)



