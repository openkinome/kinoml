from typing import List, Tuple, Union

from openeye import oechem


def create_hybrid_receptor(
    protein: oechem.OEMolBase, ligand: oechem.OEMolBase
) -> oechem.OEGraphMol:
    """
    Create a receptor for hybrid docking.

    Parameters
    ----------
    protein: oechem.OEMolBase
        An OpenEye molecule holding a prepared protein structure.
    ligand: oechem.OEMolBase
        An OpenEye molecule holding a prepared ligand structure.

    Returns
    -------
    receptor: oechem.OEGraphMol
        An OpenEye molecule holding a receptor with protein and ligand.
    """
    from openeye import oedocking

    # create receptor
    receptor = oechem.OEGraphMol()
    oedocking.OEMakeReceptor(receptor, protein, ligand)

    return receptor


def create_hint_receptor(
    protein: oechem.OEMolBase,
    hintx: Union[float, int],
    hinty: Union[float, int],
    hintz: Union[float, int],
) -> oechem.OEGraphMol:
    """
    Create a hint receptor for docking.

    Parameters
    ----------
    protein: oechem.OEMolBase
        An OpenEye molecule holding a prepared protein structure.
    hintx: float or int
        A number defining the hint x coordinate.
    hinty: float or int
        A number defining the hint y coordinate.
    hintz: float or int
        A number defining the hint z coordinate.

    Returns
    -------
    receptor: oechem.OEGraphMol
        An OpenEye molecule holding a receptor with defined binding site via hint coordinates.
    """
    from openeye import oedocking

    # create receptor
    receptor = oechem.OEGraphMol()
    oedocking.OEMakeReceptor(receptor, protein, hintx, hinty, hintz)

    return receptor


def resids_to_box(
    protein: oechem.OEMolBase, resids: List[int]
) -> Tuple[float, float, float, float, float, float]:
    """
    Retrieve box dimensions of a list if residues.

    Parameters
    ----------
    protein: oechem.OEMolBase
        An OpenEye molecule holding a protein structure.
    resids: list of int
        A list of resids defining the residues of interest.

    Returns
    -------
    box_dimensions: tuple of float
        The box dimensions in the order of xmax, ymax, zmax, xmin, ymin, zmin.
    """

    coordinates = oechem.OEFloatArray(protein.NumAtoms() * 3)
    oechem.OEGetPackedCoords(protein, coordinates)

    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    for i, atom in enumerate(protein.GetAtoms()):
        if oechem.OEAtomGetResidue(atom).GetResidueNumber() in resids:
            x_coordinates.append(coordinates[i * 3])
            y_coordinates.append(coordinates[(i * 3) + 1])
            z_coordinates.append(coordinates[(i * 3) + 2])

    box_dimensions = (
        max(x_coordinates),
        max(y_coordinates),
        max(z_coordinates),
        min(x_coordinates),
        min(y_coordinates),
        min(z_coordinates),
    )

    return box_dimensions


def create_box_receptor(
    protein: oechem.OEMolBase,
    box_dimensions: Tuple[float, float, float, float, float, float],
) -> oechem.OEGraphMol:
    """
    Create a box receptor for docking.

    Parameters
    ----------
    protein: oechem.OEMolBase
        An OpenEye molecule holding a prepared protein structure.
    box_dimensions: tuple of float
        The box dimensions in the order of xmax, ymax, zmax, xmin, ymin, zmin.

    Returns
    -------
    receptor: oechem.OEGraphMol
        An OpenEye molecule holding a receptor with defined binding site via box dimensions.
    """
    from openeye import oedocking

    # create receptor
    box = oedocking.OEBox(*box_dimensions)
    receptor = oechem.OEGraphMol()
    oedocking.OEMakeReceptor(receptor, protein, box)

    return receptor


def pose_molecules(
        receptor: oechem.OEMolBase,
        molecules: List[oechem.OEMolBase],
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Generate a binding pose of molecules in a prepared receptor with OpenEye's Posit method.

    Parameters
    ----------
    receptor: oechem.OEMolBase
        An OpenEye molecule holding the prepared receptor.
    molecules: list of oechem.OEMolBase
        A list of OpenEye molecules holding prepared molecules for docking.
    pKa_norm: bool, default=True
        Assign the predominant ionization state at pH ~7.4.

    Returns
    -------
    posed_molecules: list of oechem.OEGraphMol or None
        A list of OpenEye molecules holding the docked molecules.
    """
    from openeye import oedocking
    from ..modeling.OEModeling import generate_reasonable_conformations

    def probability(molecule: oechem.OEGraphMol):
        """Return the pose probability."""
        value = oechem.OEGetSDData(molecule, "POSIT::Probability")
        return float(value)

    # initialize receptor
    options = oedocking.OEPositOptions()
    options.SetIgnoreNitrogenStereo(True)  # nitrogen stereo centers can be problematic
    options.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
    poser = oedocking.OEPosit()
    poser.Initialize(receptor)

    posed_molecules = list()
    # pose molecules
    for molecule in molecules:
        # tautomers, enantiomers, conformations
        conformations_ensemble = generate_reasonable_conformations(
            molecule, dense=True, pKa_norm=pKa_norm
        )

        posed_conformations = list()
        for conformations in conformations_ensemble:
            result = oedocking.OESinglePoseResult()
            return_code = poser.Dock(result, conformations)
            if return_code != oedocking.OEDockingReturnCode_Success:
                # TODO: Maybe something for logging
                print(
                    f"POsing failed for molecule with title {conformations.GetTitle()} with error code "
                    f"{oedocking.OEDockingReturnCodeGetName(return_code)}."
                )
                continue
            else:
                posed_conformation = result.GetPose()

            # store probability and store pose
            oechem.OESetSDData(
                posed_conformation, "POSIT::Probability", str(result.GetProbability())
            )
            posed_conformations.append(oechem.OEGraphMol(posed_conformation))

        # sort all conformations of all tautomers and enantiomers by score
        posed_conformations.sort(key=probability, reverse=True)

        posed_molecules.append(posed_conformations[0])

    if len(posed_molecules) == 0:
        # TODO: returning None when something goes wrong
        return None

    return posed_molecules


def run_docking(
        receptor: oechem.OEMolBase,
        molecules: List[oechem.OEMolBase],
        dock_method: int,
        num_poses: int = 1,
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Dock molecules into a prepared receptor.

    Parameters
    ----------
    receptor: oechem.OEGraphMol
        An OpenEye molecule holding the prepared receptor.
    molecules: list of oechem.OEGraphMol
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
    from openeye import oedocking
    from ..modeling.OEModeling import generate_reasonable_conformations

    # initialize receptor
    dock_resolution = oedocking.OESearchResolution_High
    dock = oedocking.OEDock(dock_method, dock_resolution)
    dock.Initialize(receptor)

    def score(molecule: oechem.OEGraphMol, dock: oedocking.OEDock = dock):
        """Return the docking score."""
        value = oechem.OEGetSDData(molecule, dock.GetName())
        return float(value)

    docked_molecules = list()

    # dock molecules
    for molecule in molecules:
        # tautomers, enantiomers, conformations
        conformations_ensemble = generate_reasonable_conformations(
            molecule, dense=True, pKa_norm=pKa_norm
        )

        docked_conformations = list()
        # dock tautomers
        for conformations in conformations_ensemble:
            docked_mol = oechem.OEMol()

            # dock molecule
            return_code = dock.DockMultiConformerMolecule(docked_mol, conformations, num_poses)
            if return_code != oedocking.OEDockingReturnCode_Success:
                # TODO: Maybe something for logging
                print(
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
        hybrid_receptor: oechem.OEMolBase,
        molecules: List[oechem.OEMolBase],
        num_poses: int = 1,
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Dock molecules into a prepared receptor holding protein and ligand structure.

    Parameters
    ----------
    hybrid_receptor: oechem.OEMolBase
        An OpenEye molecule holding the prepared receptor.
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
    docked_molecules = run_docking(hybrid_receptor, molecules, dock_method, num_poses, pKa_norm)

    return docked_molecules


def chemgauss_docking(
        receptor: oechem.OEMolBase,
        molecules: List[oechem.OEMolBase],
        num_poses: int = 1,
        pKa_norm: bool = True,
) -> Union[List[oechem.OEGraphMol], None]:
    """
    Dock molecules into a prepared receptor holding a protein structure.

    Parameters
    ----------
    receptor: oechem.OEMolBase
        An OpenEye molecule holding the prepared hint or box receptor.
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
    docked_molecules = run_docking(receptor, molecules, dock_method, num_poses, pKa_norm)

    return docked_molecules
