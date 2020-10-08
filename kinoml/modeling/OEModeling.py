import logging
from typing import List, Set, Union, Iterable

from openeye import oechem, oegrid, oespruce
import pandas as pd


def read_smiles(smiles: str) -> oechem.OEGraphMol:
    """
    Read molecule from a smiles string.
    Parameters
    ----------
    smiles: str
        Smiles string.
    Returns
    -------
    molecule: oechem.OEGraphMol
        A molecule as OpenEye molecules.
    """
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles)

    molecules = []
    for molecule in ims.GetOEMols():
        molecules.append(oechem.OEGraphMol(molecule))

    return molecules[0]


def read_molecules(path: str) -> List[oechem.OEGraphMol]:
    """
    Read molecules from a file.
    Parameters
    ----------
    path: str
        Path to molecule file.
    Returns
    -------
    molecules: list of oechem.OEGraphMol
        A List of molecules as OpenEye molecules.
    """
    from pathlib import Path

    path = str(Path(path).expanduser().resolve())
    suffix = path.split(".")[-1]
    molecules = []
    with oechem.oemolistream(path) as ifs:
        if suffix == "pdb":
            ifs.SetFlavor(
                oechem.OEFormat_PDB,
                oechem.OEIFlavor_PDB_Default
                | oechem.OEIFlavor_PDB_DATA
                | oechem.OEIFlavor_PDB_ALTLOC,
            )
        # add more flavors here if behavior should be different from `Default`
        for molecule in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(molecule))

    # TODO: returns empty list if something goes wrong
    return molecules


def read_electron_density(path: str) -> Union[oegrid.OESkewGrid, None]:
    """
    Read electron density from a file.
    Parameters
    ----------
    path: str
        Path to electron density file.
    Returns
    -------
    electron_density: oegrid.OESkewGrid or None
        A List of molecules as OpenEye molecules.
    """
    from pathlib import Path

    path = str(Path(path).expanduser().resolve())
    electron_density = oegrid.OESkewGrid()
    # TODO: different map formats
    if not oegrid.OEReadMTZ(path, electron_density, oegrid.OEMTZMapType_Fwt):
        electron_density = None

    # TODO: returns None if something goes wrong
    return electron_density


def write_molecules(molecules: List[oechem.OEGraphMol], path: str):
    """
    Save molecules to file.
    Parameters
    ----------
    molecules: list of oechem.OEGraphMol
        A list of OpenEye molecules for writing.
    path: str
        File path for saving molecules.
    """
    from pathlib import Path

    path = str(Path(path).expanduser().resolve())
    with oechem.oemolostream(path) as ofs:
        for molecule in molecules:
            oechem.OEWriteMolecule(ofs, molecule)
    return


def has_ligand(molecule: oechem.OEGraphMol) -> bool:
    """
    Check if OpenEye molecule contains a ligand.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    Returns
    -------
    bool
        True if molecule has ligand, False otherwise.
    """
    ligand = oechem.OEGraphMol()
    protein = oechem.OEGraphMol()
    water = oechem.OEGraphMol()
    other = oechem.OEGraphMol()
    oechem.OESplitMolComplex(ligand, protein, water, other, molecule)

    if ligand.NumAtoms() > 0:
        return True

    return False


def select_chain(molecule, chain_id):
    """
    Select a chain from an OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    chain_id: str
        Chain identifier.
    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the selected chain.
    """
    # do not change input mol
    selection = molecule.CreateCopy()

    # delete other chains
    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if residue.GetChainID() != chain_id:
            selection.DeleteAtom(atom)

    return selection


def select_altloc(molecule, altloc_id):
    """
    Select an alternate location from an OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    altloc_id: str
        Alternate location identifier.
    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the selected alternate location.
    """
    # External libraries
    from openeye import oechem

    # do not change input mol
    selection = molecule.CreateCopy()

    allowed_altloc_ids = [" ", altloc_id]

    # delete other alternate location
    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if oechem.OEResidue.GetAlternateLocation(residue) not in allowed_altloc_ids:
            selection.DeleteAtom(atom)

    return selection


def remove_non_protein(
    molecule: oechem.OEGraphMol,
    exceptions: Union[None, List[str]] = None,
    remove_water: bool = False,
) -> oechem.OEGraphMol:
    """
    Remove non-protein atoms from an OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    exceptions: None or list of str
        Exceptions that should not be removed.
    remove_water: bool
        If water should be removed.
    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the filtered structure.
    """
    if exceptions is None:
        exceptions = []
    if remove_water is False:
        exceptions.append("HOH")

    # do not change input mol
    selection = molecule.CreateCopy()

    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if residue.IsHetAtom():
            if residue.GetName() not in exceptions:
                selection.DeleteAtom(atom)

    return selection


def _OEFixBuiltLoopFragmentNumbers(protein):
    """
    Temporary fix, thanks to Jesper!
    """
    prev_fn = -1
    # Checking for CA atoms, since this will avoid messing with the caps and built sidechains,
    # since this is only a built loop problem
    builtPred = oespruce.OEIsModeledAtom()
    for atom in protein.GetAtoms(oechem.OEIsCAlpha()):
        res = oechem.OEAtomGetResidue(atom)
        fn = res.GetFragmentNumber()
        if builtPred(atom) and prev_fn != -1:
            for ra in oechem.OEGetResidueAtoms(atom):
                r = oechem.OEAtomGetResidue(ra)
                r.SetFragmentNumber(prev_fn)
                oechem.OEAtomSetResidue(ra, r)
        else:
            prev_fn = fn


def _OEFixWaterFragmentNumbers(solvent):
    """
    Temporary fix, thanks to Jesper!
    """
    fragment_counter = {}
    for atom in solvent.GetAtoms(oechem.OEIsWater()):
        res = oechem.OEAtomGetResidue(atom)
        if res.GetInsertCode() != " ":
            continue
        if res.GetFragmentNumber() not in fragment_counter:
            fragment_counter[res.GetFragmentNumber()] = 0
        fragment_counter[res.GetFragmentNumber()] += 1
    largest_solvent_fn_count = -1
    largest_solvent_fn = -1
    for fn in fragment_counter:
        if fragment_counter[fn] > largest_solvent_fn_count:
            largest_solvent_fn_count = fragment_counter[fn]
            largest_solvent_fn = fn
    if largest_solvent_fn < 0:
        return
    for atom in solvent.GetAtoms(oechem.OEIsWater(True)):
        res = oechem.OEAtomGetResidue(atom)
        res.SetFragmentNumber(largest_solvent_fn)
        oechem.OEAtomSetResidue(atom, res)


def _OEFixConnectionNH(protein):
    """
    Temporary fix, thanks to Jesper!
    """
    for atom in protein.GetAtoms(
        oechem.OEAndAtom(oespruce.OEIsModeledAtom(), oechem.OEIsNitrogen())
    ):
        if oechem.OEGetPDBAtomIndex(atom) == oechem.OEPDBAtomName_N:
            expected_h_count = 1
            if oechem.OEGetResidueIndex(atom) == oechem.OEResidueIndex_PRO:
                expected_h_count = 0
            if atom.GetTotalHCount() != expected_h_count:
                oechem.OESuppressHydrogens(atom)
                atom.SetImplicitHCount(1)
                oechem.OEAddExplicitHydrogens(protein, atom)
                for nbr in atom.GetAtoms(oechem.OEIsHydrogen()):
                    oechem.OESet3DHydrogenGeom(protein, nbr)


def _OEFixLoopIssues(du):
    """
    Temporary fix, thanks to Jesper!
    """
    impl = du.GetImpl()
    protein = impl.GetProtein()

    _OEFixBuiltLoopFragmentNumbers(protein)
    _OEFixConnectionNH(protein)

    oechem.OEPDBOrderAtoms(protein)

    solvent = impl.GetSolvent()
    _OEFixWaterFragmentNumbers(solvent)


def _OEFixMissingOXT(du):
    """
    Temporary fix, thanks to Jesper!
    """
    impl = du.GetImpl()
    protein = impl.GetProtein()

    for atom in protein.GetAtoms():
        if "H'" in atom.GetName():
            atom.SetAtomicNum(8)
            atom.SetName("OXT")
            atom.SetFormalCharge(-1)


def _prepare_structure(
    structure: oechem.OEGraphMol,
    has_ligand: bool = False,
    electron_density: Union[oegrid.OESkewGrid, None] = None,
    loop_db: Union[str, None] = None,
    ligand_name: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> Union[oechem.OEDesignUnit, None]:
    """
    Prepare an OpenEye molecule holding a protein ligand complex for docking.
    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding a structure with protein and optionally a ligand.
    has_ligand: bool
        If structure contains a ligand that should be used in design unit generation.
    electron_density: oegrid.OESkewGrid
        An OpenEye grid holding the electron density.
    loop_db: str or None
        Path to OpenEye Spruce loop database.
    ligand_name: str or None
        The name of the ligand located in the binding pocket of interest.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.
    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.
    """

    def _has_residue_number(atom, residue_numbers=real_termini):
        """Return True if atom matches any given residue number."""
        residue = oechem.OEAtomGetResidue(atom)
        return any(
            [
                True if residue.GetResidueNumber() == residue_number else False
                for residue_number in residue_numbers
            ]
        )

    # set design unit options
    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    if cap_termini is False:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    if loop_db is not None:
        from pathlib import Path

        loop_db = str(Path(loop_db).expanduser().resolve())
        design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    # cap all termini but biologically real termini
    if real_termini is not None and cap_termini is True:
        oespruce.OECapTermini(structure, oechem.PyAtomPredicate(_has_residue_number))
        # already capped, preserve biologically real termini
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)

    # make design units
    if has_ligand:
        if electron_density is None:
            design_units = list(
                oespruce.OEMakeDesignUnits(
                    structure, structure_metadata, design_unit_options
                )
            )
            # filter design units for ligand of interest
            if ligand_name is not None:
                design_units = [
                    design_unit
                    for design_unit in design_units
                    if ligand_name in design_unit.GetTitle()
                ]

        else:
            design_units = list(
                oespruce.OEMakeDesignUnits(
                    structure, electron_density, structure_metadata, design_unit_options
                )
            )
    else:
        design_units = list(
            oespruce.OEMakeBioDesignUnits(
                structure, structure_metadata, design_unit_options
            )
        )

    if len(design_units) >= 1:
        design_unit = design_units[0]
    else:
        # TODO: Returns None if something goes wrong
        return None

    # fix loop issues and missing OXT
    _OEFixLoopIssues(design_unit)
    _OEFixMissingOXT(design_unit)

    return design_unit


def prepare_complex(
    protein_ligand_complex: oechem.OEGraphMol,
    electron_density: Union[oegrid.OESkewGrid, None] = None,
    loop_db: Union[str, None] = None,
    ligand_name: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> Union[oechem.OEDesignUnit, None]:
    """
    Prepare an OpenEye molecule holding a protein ligand complex for docking.
    Parameters
    ----------
    protein_ligand_complex: oechem.OEGraphMol
        An OpenEye molecule holding a structure with protein and ligand.
    electron_density: oegrid.OESkewGrid
        An OpenEye grid holding the electron density.
    loop_db: str or None
        Path to OpenEye Spruce loop database.
    ligand_name: str or None
        The name of the ligand located in the binding pocket of interest.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.
    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.
    """
    return _prepare_structure(
        structure=protein_ligand_complex,
        has_ligand=True,
        electron_density=electron_density,
        loop_db=loop_db,
        ligand_name=ligand_name,
        cap_termini=cap_termini,
        real_termini=real_termini,
    )


def prepare_protein(
    protein: oechem.OEGraphMol,
    loop_db: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> Union[oechem.OEDesignUnit, None]:
    """
    Prepare an OpenEye molecule holding a protein structure for docking.
    Parameters
    ----------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a structure with protein.
    loop_db: str
        Path to OpenEye Spruce loop database.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.
    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.
    """
    return _prepare_structure(
        structure=protein,
        loop_db=loop_db,
        cap_termini=cap_termini,
        real_termini=real_termini,
    )


def klifs_kinases_by_uniprot_id(uniprot_id: str) -> pd.DataFrame:
    """
    Retrieve KLIFS structure details about kinases matching the given Uniprot ID.
    Parameters
    ----------
    uniprot_id: str
        Uniprot identifier.
    Returns
    -------
    kinases: pd.DataFrame
        KLIFS structure details.
    """
    import klifs_utils

    kinase_names = klifs_utils.remote.kinases.kinase_names()
    kinases = klifs_utils.remote.kinases.kinases_from_kinase_names(
        list(kinase_names.name)
    )
    kinase_id = kinases[kinases.uniprot == uniprot_id].kinase_ID.iloc[0]
    kinases = klifs_utils.remote.structures.structures_from_kinase_ids([kinase_id])
    return kinases


def klifs_kinase_from_uniprot_id(uniprot_id: str) -> pd.DataFrame:
    """
    Retrieve KLIFS kinase details about the kinase matching the given Uniprot ID.
    Parameters
    ----------
    uniprot_id: str
        Uniprot identifier.
    Returns
    -------
    kinases: pd.Series
        KLIFS structure details.
    """
    import klifs_utils

    kinase_names = klifs_utils.remote.kinases.kinase_names()
    kinases = klifs_utils.remote.kinases.kinases_from_kinase_names(
        list(kinase_names.name)
    )
    kinase = kinases[kinases.uniprot == uniprot_id].iloc[0]
    return kinase


def get_klifs_ligand(structure_id: int) -> oechem.OEGraphMol:
    """
    Retrieve orthosteric ligand from KLIFS.
    Parameters
    ----------
    structure_id: int
        KLIFS structure identifier.
    Returns
    -------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding the orthosteric ligand.
    """
    import klifs_utils
    from ..utils import LocalFileStorage

    file_path = LocalFileStorage.klifs_ligand_mol2(structure_id)

    if not file_path.is_file():
        mol2_text = klifs_utils.remote.coordinates.ligand._ligand_mol2_text(
            structure_id
        )
        with open(file_path, "w") as wf:
            wf.write(mol2_text)

    molecule = read_molecules(file_path)[0]

    return molecule


def generate_tautomers(molecule: oechem.OEGraphMol) -> List[oechem.OEGraphMol]:
    """
    Generate reasonable tautomers of a given molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    Returns
    -------
    tautomers: list of oechem.OEGraphMol
        A list of OpenEye molecules holding the tautomers.
    """
    from openeye import oechem, oequacpac

    tautomer_options = oequacpac.OETautomerOptions()
    tautomer_options.SetMaxTautomersGenerated(4096)
    tautomer_options.SetMaxTautomersToReturn(16)
    tautomer_options.SetCarbonHybridization(True)
    tautomer_options.SetMaxZoneSize(50)
    tautomer_options.SetApplyWarts(True)
    pKa_norm = True
    tautomers = [
        oechem.OEGraphMol(tautomer)
        for tautomer in oequacpac.OEGetReasonableTautomers(
            molecule, tautomer_options, pKa_norm
        )
    ]
    return tautomers


def generate_enantiomers(
    molecule: oechem.OEGraphMol, max_centers: int = 12, ignore: bool = False
) -> List[oechem.OEGraphMol]:
    """
    Generate enantiomers of a given molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    max_centers: int
        The maximal number of stereo centers to enumerate.
    ignore: bool
        If specified stereo centers should be ignored.
    Returns
    -------
    enantiomers: list of oechem.OEGraphMol
        A list of OpenEye molecules holding the enantiomers.
    """
    from openeye import oechem, oeomega

    enantiomers = [
        oechem.OEGraphMol(enantiomer)
        for enantiomer in oeomega.OEFlipper(molecule, max_centers, ignore)
    ]
    return enantiomers


def generate_conformations(
    molecule: oechem.OEGraphMol, max_conformations: int = 1000
) -> oechem.OEMol:
    """
    Generate conformations of a given molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    max_conformations: int
        Maximal number of conformations to generate.
    Returns
    -------
    conformations: oechem.OEMol
        An OpenEye multi-conformer molecule holding the generated conformations.
    """
    from openeye import oechem, oeomega

    omega_options = oeomega.OEOmegaOptions()
    omega_options.SetMaxSearchTime(60.0)  # time out
    omega_options.SetMaxConfs(max_conformations)
    omega = oeomega.OEOmega(omega_options)
    omega.SetStrictStereo(False)
    conformations = oechem.OEMol(molecule)
    omega.Build(conformations)
    return conformations


def generate_reasonable_conformations(
    molecule: oechem.OEGraphMol,
) -> List[oechem.OEMol]:
    """
    Generate conformations of reasonable enantiomers and tautomers of a given molecule.
    Parameters
    ----------
    molecule: oechem.ORGraphMol
        An OpenEye molecule.
    Returns
    -------
    conformations_ensemble: list of oechem.OEMol
        A list of multi-conformer molecules.
    """
    import itertools

    tautomers = generate_tautomers(molecule)
    enantiomers = [generate_enantiomers(tautomer) for tautomer in tautomers]
    conformations_ensemble = [
        generate_conformations(enantiomer)
        for enantiomer in itertools.chain.from_iterable(enantiomers)
    ]
    return conformations_ensemble


def optimize_poses(
    docking_poses: List[oechem.OEGraphMol],
    protein: Union[oechem.OEMolBase, oechem.OEGraphMol],
) -> List[oechem.OEGraphMol]:
    """
    Optimize the torsions of docking poses in a protein binding site.
    Parameters
    ----------
    docking_poses: list of oechem.OEGraphMol
        The docking poses to optimize.
    protein: oechem.OEGraphMol or oechem.MolBase
        The OpenEye molecule holding a protein structure.
    Returns
    -------
    optimized_docking_poses: list of oechem.OEGraphMol
        The optimized docking poses.
    """
    from openeye import oeszybki

    options = oeszybki.OESzybkiOptions()
    options.SetRunType(oeszybki.OERunType_TorsionsOpt)
    options.GetProteinOptions().SetExactVdWProteinLigand(True)
    options.GetProteinOptions().SetProteinElectrostaticModel(
        oeszybki.OEProteinElectrostatics_ExactCoulomb
    )
    options.GetOptOptions().SetGradTolerance(0.00001)
    szybki = oeszybki.OESzybki(options)
    szybki.SetProtein(protein)

    optimized_docking_poses = []
    for docking_pose in docking_poses:
        result = oeszybki.OESzybkiResults()
        szybki(docking_pose, result)
        optimized_docking_poses.append(oechem.OEGraphMol(docking_pose))

    return optimized_docking_poses


def overlay_molecules(
    reference_molecule: oechem.OEGraphMol,
    fit_molecule: oechem.OEMol,
    return_overlay: bool = True,
) -> (int, List[oechem.OEGraphMol]):
    """
    Overlay two molecules and calculate TanimotoCombo score.
    Parameters
    ----------
    reference_molecule: oechem.OEGraphMol
        An OpenEye molecule holding the reference molecule for overlay.
    fit_molecule: oechem.OEMol
        An OpenEye multi-conformer molecule holding the fit molecule for overlay.
    return_overlay: bool
        If the best scored overlay of molecules should be returned.
    Returns
    -------
        : int or int and list of oechem.OEGraphMol
        The TanimotoCombo score of the best overlay and the overlay if score_only is set False.
    """
    from openeye import oechem, oeshape

    prep = oeshape.OEOverlapPrep()
    prep.Prep(reference_molecule)

    overlay = oeshape.OEOverlay()
    overlay.SetupRef(reference_molecule)

    prep.Prep(fit_molecule)
    score = oeshape.OEBestOverlayScore()
    overlay.BestOverlay(score, fit_molecule, oeshape.OEHighestTanimoto())
    if not return_overlay:
        return score.GetTanimotoCombo()
    else:
        overlay = [reference_molecule]
        fit_molecule = oechem.OEGraphMol(
            fit_molecule.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx()))
        )
        score.Transform(fit_molecule)
        overlay.append(fit_molecule)
        return score.GetTanimotoCombo(), overlay


def generate_isomeric_smiles_representations(molecule: oechem.OEGraphMol) -> Set[str]:
    """
    Generate reasonable isomeric smiles of a given OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    Returns
    -------
    smiles_set: set of str
        A set of reasonable isomeric smiles strings.
    """
    import itertools

    tautomers = generate_tautomers(molecule)
    enantiomers = [generate_enantiomers(tautomer) for tautomer in tautomers]
    smiles_set = set(
        [
            oechem.OEMolToSmiles(enantiomer)
            for enantiomer in itertools.chain.from_iterable(enantiomers)
        ]
    )
    return smiles_set


def compare_molecules(
    molecule1: oechem.OEGraphMol, molecule2: oechem.OEGraphMol
) -> bool:
    """
    Compare two OpenEye molecules.
    Parameters
    ----------
    molecule1: oechem.OEGraphMol
        The first OpenEye molecule.
    molecule2: oechem.OEGraphMol
        The second OpenEye molecule.
    Returns
    -------
    : bool
        True if same molecules, else False.
    """
    reasonable_isomeric_smiles1 = generate_isomeric_smiles_representations(molecule1)
    reasonable_isomeric_smiles2 = generate_isomeric_smiles_representations(molecule2)

    if len(reasonable_isomeric_smiles1 & reasonable_isomeric_smiles2) == 0:
        return False
    else:
        return True


def string_similarity(string1: str, string2: str) -> float:
    """
    Compare the characters of two strings.
    Parameters
    ----------
    string1: str
        The first string.
    string2: str
        The second string.
    Returns
    -------
        : float
        Similarity of strings expressed as fraction of matching characters.
    """
    common = len([x for x, y in zip(string1, string2) if x == y])
    return common / max([len(string1), len(string2)])


def smiles_from_pdb(ligand_ids: Iterable[str]) -> dict:
    """
    Retrieve SMILES of molecules defined by their PDB chemical identifier.
    Parameters
    ----------
    ligand_ids: iterable of str
        Iterable of PDB chemical identifiers.
    Returns
    -------
    ligands: dict
        Dictionary with PDB chemical identifier as keys and SMILES as values.
    """
    import requests
    from xml.etree import ElementTree

    ligands = {}
    url = f"https://www.rcsb.org/pdb/rest/describeHet?chemicalID={','.join(set(ligand_ids))}"
    response = requests.get(url)
    tree = ElementTree.fromstring(response.text)
    for element in tree.findall(".//ligand"):
        ligand_id = element.get("chemicalID")
        if element.find("smiles") is not None:
            smiles = element.find("smiles").text
        else:
            smiles = None
        ligands[ligand_id] = smiles
    return ligands


def get_sequence(structure: oechem.OEGraphMol) -> str:
    """
    Get the amino acid sequence with one letter characters of an OpenEye molecule holding a protein structure. All
    residues not perceived as amino acid will receive the character 'X'.
    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure.
    Returns
    -------
    sequence: str
        The amino acid sequence of the protein with one letter characters.
    """
    sequence = []
    hv = oechem.OEHierView(structure)
    for residue in hv.GetResidues():
        if oechem.OEIsStandardProteinResidue(residue):
            sequence.append(
                oechem.OEGetAminoAcidCode(
                    oechem.OEGetResidueIndex(residue.GetResidueName())
                )
            )
        else:
            sequence.append("X")
    sequence = "".join(sequence)
    return sequence


def mutate_structure(
    target_structure: oechem.OEGraphMol, template_sequence: str
) -> oechem.OEGraphMol:
    """
    Mutate a protein structure according to an amino acid sequence.
    Parameters
    ----------
    target_structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure to mutate.
    template_sequence: str
        A template one letter amino acid sequence, which defines the sequence the target structure should be mutated
        to. Protein residues not matching a template sequence will be either mutated or deleted.
    Returns
    -------
    mutated_structure: oechem.OEGraphMol
        An OpenEye molecule holding the mutated protein structure.
    """
    from Bio import pairwise2

    # the hierarchy view is more stable if reinitialized after each change
    # https://docs.eyesopen.com/toolkits/python/oechemtk/biopolymers.html#a-hierarchy-view
    finished = False
    while not finished:
        altered = False
        # align template and target sequences
        target_sequence = get_sequence(target_structure)
        template_sequence_aligned, target_sequence_aligned = pairwise2.align.globalxs(
            template_sequence, target_sequence, -10, 0
        )[0][:2]
        logging.debug(f"Template sequence:\n{template_sequence}")
        logging.debug(f"Target sequence:\n{target_sequence}")
        hierview = oechem.OEHierView(target_structure)
        structure_residues = hierview.GetResidues()
        # adjust target structure to match template sequence
        for template_sequence_residue, target_sequence_residue in zip(
            template_sequence_aligned, target_sequence_aligned
        ):
            if template_sequence_residue == "-":
                # delete any non protein residue from target structure
                structure_residue = structure_residues.next()
                if target_sequence_residue != "X":
                    # delete
                    for atom in structure_residue.GetAtoms():
                        target_structure.DeleteAtom(atom)
                    # break loop and reinitialize
                    altered = True
                    break
            else:
                # compare amino acids
                if target_sequence_residue != "-":
                    structure_residue = structure_residues.next()
                    if target_sequence_residue not in ["X", template_sequence_residue]:
                        # mutate
                        structure_residue = structure_residue.GetOEResidue()
                        three_letter_code = oechem.OEGetResidueName(
                            oechem.OEGetResidueIndexFromCode(template_sequence_residue)
                        )
                        oespruce.OEMutateResidue(
                            target_structure, structure_residue, three_letter_code
                        )
                        # break loop and reinitialize
                        altered = True
                        break
        # leave while loop if no changes were introduced
        if not altered:
            finished = True
    # OEMutateResidue doesn't build sidechains and doesn't add hydrogens automatically
    oespruce.OEBuildSidechains(target_structure)
    oechem.OEPlaceHydrogens(target_structure)
    # update residue information
    oechem.OEPerceiveResidues(target_structure, oechem.OEPreserveResInfo_All)

    return target_structure


def renumber_structure(
    target_structure: oechem.OEGraphMol, residue_numbers: List[int]
) -> oechem.OEGraphMol:
    """
    Renumber the residues of a protein structure according to the given list of residue numbers.
    Parameters
    ----------
    target_structure: oechem.OEGraphMol
        An OpenEye molecule holding the protein structure to renumber.
    residue_numbers: list of int
        A list of residue numbers matching the order of the target structure.
    Returns
    -------
    renumbered_structure: oechem.OEGraphMol
        An OpenEye molecule holding the cropped protein structure.
    """
    import copy

    renumbered_structure = copy.deepcopy(
        target_structure
    )  # don't touch input structure
    hierview = oechem.OEHierView(renumbered_structure)
    structure_residues = hierview.GetResidues()
    for residue_number, structure_residue in zip(residue_numbers, structure_residues):
        structure_residue_mod = structure_residue.GetOEResidue()
        structure_residue_mod.SetResidueNumber(residue_number)
        for residue_atom in structure_residue.GetAtoms():
            oechem.OEAtomSetResidue(residue_atom, structure_residue_mod)

    return renumbered_structure


def superpose_proteins(
    reference_protein: oechem.OEGraphMol, fit_protein: oechem.OEGraphMol
) -> oechem.OEGraphMol:
    """
    Superpose a protein structure onto a reference protein.
    Parameters
    ----------
    reference_protein: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure which will be used as reference during superposition.
    fit_protein: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure which will be superposed onto the reference protein.
    Returns
    -------
    superposed_protein: oechem.OEGraphMol
        An OpenEye molecule holding the superposed protein structure.
    """
    # do not modify input
    superposed_protein = fit_protein.CreateCopy()

    # set superposition method
    options = oespruce.OESuperpositionOptions()
    options.SetSuperpositionType(oespruce.OESuperpositionType_Global)

    # perform superposition
    superposition = oespruce.OEStructuralSuperposition(
        reference_protein, superposed_protein, options
    )
    superposition.Transform(superposed_protein)

    return superposed_protein


def update_residue_identifiers(
    structure: oechem.OEGraphMol, keep_protein_residue_ids: bool = True
) -> oechem.OEGraphMol:
    """
    Updates the atom, residue and chain ids of the given molecular structure. All residues become part of chain A. Atom
    ids will start from 1. Residue will start from 1, except protein residue ids are fixed. This is especially useful,
    if molecules were merged, which can result in overlapping atom and residue ids as well as separate chains.
    Parameters
    ----------
    structure: oechem.OEGraphMol
        The OpenEye molecule structure for updating atom and residue ids.
    keep_protein_residue_ids: bool
        If the protein residues should be kept.
    Returns
    -------
    structure: oechem.OEGraphMol
        The OpenEye molecule structure with updated atom and residue ids.
    """
    # update residue ids
    residue_number = 0
    hierarchical_view = oechem.OEHierView(structure)
    for hv_residue in hierarchical_view.GetResidues():
        residue = hv_residue.GetOEResidue()
        residue.SetChainID("A")
        if not residue.IsHetAtom() and keep_protein_residue_ids:
            if (
                residue.GetName() == "NME"
                and residue.GetResidueNumber() == residue_number
            ):
                # NME residues may have same id as preceding residue
                residue_number += 1
            else:
                # catch protein residue id if those should not be touched
                residue_number = residue.GetResidueNumber()

        else:
            # change residue id
            residue_number += 1
        residue.SetResidueNumber(residue_number)
        for atom in hv_residue.GetAtoms():
            oechem.OEAtomSetResidue(atom, residue)

    # update residue identifiers, except atom names, residue ids,
    # residue names, fragment number, chain id and record type
    preserved_info = (
        oechem.OEPreserveResInfo_ResidueNumber
        | oechem.OEPreserveResInfo_ResidueName
        | oechem.OEPreserveResInfo_HetAtom
        | oechem.OEPreserveResInfo_AtomName
        | oechem.OEPreserveResInfo_FragmentNumber
        | oechem.OEPreserveResInfo_ChainID
    )
    oechem.OEPerceiveResidues(structure, preserved_info)

    return structure


def clashing_atoms(molecule1: oechem.OEGraphMol, molecule2: oechem.OEGraphMol) -> bool:
    """
    Evaluates if the atoms of two molecules are clashing.
    Parameters
    ----------
    molecule1: oechem.OEGraphMol
        An OpenEye molecule.
    molecule2: oechem.OEGraphMol
        An OpenEye molecule.
    Returns
    -------
    : bool
        If any atoms of two molecules are clashing
    """
    import math

    oechem.OEAssignCovalentRadii(molecule1)
    oechem.OEAssignCovalentRadii(molecule2)
    coordinates1_list = [
        molecule1.GetCoords()[x] for x in sorted(molecule1.GetCoords().keys())
    ]
    coordinates2_list = [
        molecule2.GetCoords()[x] for x in sorted(molecule2.GetCoords().keys())
    ]
    for atom1, coordinates1 in zip(molecule1.GetAtoms(), coordinates1_list):
        for atom2, coordinates2 in zip(molecule2.GetAtoms(), coordinates2_list):
            clash_threshold = atom1.GetRadius() + atom2.GetRadius()
            distance = math.sqrt(
                sum(
                    (coordinate1 - coordinate2) ** 2.0
                    for coordinate1, coordinate2 in zip(coordinates1, coordinates2)
                )
            )
            if distance <= clash_threshold:
                return True
    return False
