import logging
from typing import List, Union

from openeye import oechem


def resids_to_box_molecule(protein: oechem.OEMolBase, resids: List[int]) -> oechem.OEGraphMol:
    """
    Retrieve a box molecule spanning the given protein residue IDs.

    Parameters
    ----------
    protein: oechem.OEMolBase
        An OpenEye molecule holding a protein structure.
    resids: list of int
        A list of resids defining the residues of interest.

    Returns
    -------
    box_molecule: oechem.OEGraphMol
        Rectangular box molecule spanning the region of the given protein defined by the given
        residue IDs.
    """
    from openeye import oedocking

    coordinates = oechem.OEFloatArray(protein.NumAtoms() * 3)
    oechem.OEGetPackedCoords(protein, coordinates)

    # collect coordinates
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    for i, atom in enumerate(protein.GetAtoms()):
        if oechem.OEAtomGetResidue(atom).GetResidueNumber() in resids:
            x_coordinates.append(coordinates[i * 3])
            y_coordinates.append(coordinates[(i * 3) + 1])
            z_coordinates.append(coordinates[(i * 3) + 2])

    if any([
        len(coordinates) == 0 for coordinates in [x_coordinates, y_coordinates, z_coordinates]
    ]):
        raise ValueError("Given residue IDs do not match any residue in the given protein.")

    # calculate box dimensions
    box_dimensions = (
        max(x_coordinates),
        max(y_coordinates),
        max(z_coordinates),
        min(x_coordinates),
        min(y_coordinates),
        min(z_coordinates),
    )

    # create box
    box = oechem.OEBox()
    box.Setup(*box_dimensions)

    # make box molecule
    box_molecule = oechem.OEGraphMol()
    oedocking.OEMakeBoxMolecule(box_molecule, box)

    return box_molecule


def pose_molecules(
        design_unit: oechem.OEDesignUnit,
        molecules: List[oechem.OEMolBase],
        pKa_norm: bool = True,
        score_pose: bool = False,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Generate a binding pose of molecules in a prepared receptor with OpenEye's Posit method.

    Parameters
    ----------
    design_unit: oechem. OEDesignUnit
        A design unit with a receptor object.
    molecules: list of oechem.OEMolBase
        A list of OpenEye molecules holding prepared molecules for docking.
    pKa_norm: bool, default=True
        Assign the predominant ionization state at pH ~7.4.
    score_pose: bool, default=False
        Score the best docking pose per ligand and add the proper SD tag.

    Returns
    -------
    posed_molecules: list of oechem.OEGraphMol or None
        A list of OpenEye molecules holding the docked molecules.
    """
    from openeye import oedocking, oeomega
    from ..modeling.OEModeling import generate_reasonable_conformations

    def probability(molecule: oechem.OEGraphMol):
        """Return the pose probability."""
        value = oechem.OEGetSDData(molecule, "POSIT::Probability")
        return float(value)

    # initialize receptor
    options = oedocking.OEPositOptions()
    options.SetIgnoreNitrogenStereo(True)  # nitrogen stereo centers can be problematic
    options.SetPoseRelaxMode(
        oedocking.OEPoseRelaxMode_NONE
    )  # relaxation is slow and would also affect the protein, which is currently not returned
    poser = oedocking.OEPosit(options)
    poser.AddReceptor(design_unit)

    posed_molecules = list()
    # pose molecules
    for molecule in molecules:
        # tautomers, enantiomers, conformations
        conformations_ensemble = generate_reasonable_conformations(
            molecule,
            oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose),
            pKa_norm=pKa_norm
        )

        posed_conformations = list()
        for conformations in conformations_ensemble:
            result = oedocking.OESinglePoseResult()
            return_code = poser.Dock(result, conformations)
            if return_code != oedocking.OEDockingReturnCode_Success:
                logging.debug(
                    f"Posing failed for molecule with title {conformations.GetTitle()} with error code "
                    f"{oedocking.OEDockingReturnCodeGetName(return_code)}."
                )
                continue
            else:
                posed_conformation = result.GetPose()

                # store probability and pose
                oechem.OESetSDData(
                    posed_conformation, "POSIT::Probability", str(result.GetProbability())
                )
                posed_conformations.append(oechem.OEGraphMol(posed_conformation))

        # sort all conformations of all tautomers and enantiomers by score
        posed_conformations.sort(key=probability, reverse=True)

        # keep conformation with highest probability
        if len(posed_conformations) > 0:
            best_pose = posed_conformations[0]
            if score_pose:
                # calculate and store ChemGauss4 docking score
                pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
                pose_scorer.Initialize(design_unit)
                pose_scorer.ScoreLigand(best_pose)
                oedocking.OESetSDScore(best_pose, pose_scorer, pose_scorer.GetName())
            posed_molecules.append(best_pose)

    if len(posed_molecules) == 0:
        # TODO: returning None when something goes wrong
        return None

    return posed_molecules


def run_docking(
        design_unit: oechem.OEDesignUnit,
        molecules: List[oechem.OEMolBase],
        dock_method: int,
        num_poses: int = 1,
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Dock molecules into a prepared design unit containing a receptor object.

    Parameters
    ----------
    design_unit: oechem.OEDesignUnit
        A design unit with a receptor object.
    molecules: list of oechem.OEMolBase
        A list of OpenEye molecules holding prepared molecules for docking.
    dock_method: int
        Constant defining the docking method.
    num_poses: int
        Number of docking poses to generate per molecule.
    pKa_norm: bool, default=True
        Assign the predominant ionization state at pH ~7.4.

    Returns
    -------
    docked_molecules: list of oechem.OEGraphMol or None
        A list of OpenEye molecules holding the docked molecules.
    """
    from openeye import oedocking, oeomega
    from ..modeling.OEModeling import generate_reasonable_conformations

    # initialize receptor
    dock_resolution = oedocking.OESearchResolution_High
    dock = oedocking.OEDock(dock_method, dock_resolution)
    dock.Initialize(design_unit)

    def score(molecule: oechem.OEGraphMol, dock: oedocking.OEDock = dock):
        """Return the docking score."""
        value = oechem.OEGetSDData(molecule, dock.GetName())
        return float(value)

    docked_molecules = list()

    # dock molecules
    for molecule in molecules:
        # tautomers, enantiomers, conformations
        conformations_ensemble = generate_reasonable_conformations(
            molecule,
            options=oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose),
            pKa_norm=pKa_norm
        )

        docked_conformations = list()
        # dock tautomers
        for conformations in conformations_ensemble:
            docked_mol = oechem.OEMol()

            # dock molecule
            return_code = dock.DockMultiConformerMolecule(docked_mol, conformations, num_poses)
            if return_code != oedocking.OEDockingReturnCode_Success:
                logging.debug(
                    f"Docking failed for molecule with title {conformations.GetTitle()} with error code "
                    f"{oedocking.OEDockingReturnCodeGetName(return_code)}."
                )
                continue

            # store docking data
            oedocking.OESetSDScore(docked_mol, dock, dock.GetName())

            # expand conformations
            for conformation in docked_mol.GetConfs():
                docked_conformations.append(oechem.OEGraphMol(conformation))

        # sort all conformations of all tautomers and enantiomers by score
        docked_conformations.sort(key=score)

        # keep number of conformations as specified by num_poses
        docked_molecules += docked_conformations[:num_poses]

    if len(docked_molecules) == 0:
        # TODO: returning None when something goes wrong
        return None

    return docked_molecules


def hybrid_docking(
        design_unit: oechem.OEDesignUnit,
        molecules: List[oechem.OEMolBase],
        num_poses: int = 1,
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Dock molecules into a prepared design unit containing a hybrid receptor object.

    Parameters
    ----------
    design_unit: oechem.OEDesignUnit
        A design unit with a hybrid receptor object.
    molecules: list of oechem.OEMolBase
        A list of OpenEye molecules holding prepared molecules for docking.
    num_poses: int
        Number of docking poses to generate per molecule.
    pKa_norm: bool, default=True
        Assign the predominant ionization state at pH ~7.4.

    Returns
    -------
    docked_molecules: list of oechem.OEGraphMol or None
        A list of OpenEye molecules holding the docked molecules.
    """
    from openeye import oedocking

    dock_method = oedocking.OEDockMethod_Hybrid2
    docked_molecules = run_docking(design_unit, molecules, dock_method, num_poses, pKa_norm)

    return docked_molecules


def fred_docking(
        design_unit: oechem.OEDesignUnit,
        molecules: List[oechem.OEMolBase],
        num_poses: int = 1,
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Dock molecules into a prepared design unit containing a receptor object.

    Parameters
    ----------
    design_unit: oechem.OEDesignUnit
        A design unit with a receptor object.
    molecules: list of oechem.OEMolBase
        A list of OpenEye molecules holding prepared molecules for docking.
    num_poses: int
        Number of docking poses to generate per molecule.
    pKa_norm: bool, default=True
        Assign the predominant ionization state at pH ~7.4.

    Returns
    -------
    docked_molecules: list of oechem.OEGraphMol or None
        A list of OpenEye molecules holding the docked molecules.
    """
    from openeye import oedocking

    dock_method = oedocking.OEDockMethod_Chemgauss4
    docked_molecules = run_docking(design_unit, molecules, dock_method, num_poses, pKa_norm)

    return docked_molecules
