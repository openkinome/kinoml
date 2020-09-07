from typing import List, Union

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


def _prepare_structure(
    structure: oechem.OEGraphMol,
    has_ligand: bool,
    electron_density: Union[oegrid.OESkewGrid, None] = None,
    loop_db: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> Union[List[Union[oechem.OEGraphMol, None]], oechem.OEGraphMol, None]:
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
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.
    Returns
    -------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a prepared protein structure.
    ligand: oechem.OEGraphMol
        An OpenEye molecule holding a prepared ligand structure.
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

    if len(design_units) == 1:
        design_unit = design_units[0]
    elif len(design_units) > 1:
        design_unit = design_units[0]
    else:
        # TODO: Returns None or list of Nones if something goes wrong
        if has_ligand:
            return [None, None]
        else:
            return None

    # get protein
    protein = oechem.OEGraphMol()
    design_unit.GetProtein(protein)

    # add missing OXT backbone atoms, not handled by OEFixBackbone in OESpruce 1.1.0
    for atom in protein.GetAtoms():
        if "H'" in atom.GetName():
            atom.SetAtomicNum(8)
            atom.SetName("OXT")
            atom.SetFormalCharge(-1)
    if has_ligand:
        # get ligand
        ligand = oechem.OEGraphMol()
        design_unit.GetLigand(ligand)
        return [protein, ligand]
    else:
        return protein


def prepare_complex(
    protein_ligand_complex: oechem.OEGraphMol,
    electron_density: Union[oegrid.OESkewGrid, None] = None,
    loop_db: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> List[Union[oechem.OEGraphMol, None]]:
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
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.
    Returns
    -------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a prepared protein structure.
    ligand: oechem.OEGraphMol
        An OpenEye molecule holding a prepared ligand structure.
    """
    return _prepare_structure(
        structure=protein_ligand_complex,
        has_ligand=True,
        electron_density=electron_density,
        loop_db=loop_db,
        cap_termini=cap_termini,
        real_termini=real_termini,
    )


def prepare_protein(
    protein: oechem.OEGraphMol,
    loop_db: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> Union[oechem.OEGraphMol, None]:
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
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a prepared protein structure.
    """
    return _prepare_structure(
        structure=protein,
        has_ligand=False,
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
    from openeye import oechem

    mol2_text = klifs_utils.remote.coordinates.ligand._ligand_mol2_text(structure_id)
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_MOL2)
    ims.openstring(mol2_text)

    molecules = []
    for molecule in ims.GetOEGraphMols():
        molecules.append(oechem.OEGraphMol(molecule))

    return molecules[0]


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


def generate_conformers(
    molecule: oechem.OEGraphMol, max_conformations: int = 1000
) -> oechem.OEMol:
    """
    Generate enantiomers of a given molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    max_conformations: int
        Maximal number of conformations to generate.
    Returns
    -------
    conformers: oechem.OEMol
        An OpenEye multi-conformer molecule holding the generated conformations.
    """
    from openeye import oechem, oeomega

    omega_options = oeomega.OEOmegaOptions()
    omega_options.SetMaxSearchTime(60.0)  # time out
    omega_options.SetMaxConfs(max_conformations)
    omega = oeomega.OEOmega(omega_options)
    omega.SetStrictStereo(False)
    conformers = oechem.OEMol(molecule)
    omega.Build(conformers)
    return conformers


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


def select_structure(uniprot_id: str, smiles: str) -> Union[None, pd.Series]:
    """
    Select a suitable kinase structure for docking a small molecule into the orthosteric pocket.
    Parameters
    ----------
    uniprot_id: str
        Uniprot identifier.
    smiles: str
        The molecule in smiles format.
    Returns
    -------
        : pd.Series
        Details about most reasonable kinase structure for docking the small molecule.
    """
    import itertools

    # search for available kinase structure
    # TODO: ligand might be co-crystalized with kinase of different species but very similar sequence
    kinases = klifs_kinases_by_uniprot_id(uniprot_id)
    if len(kinases) == 0:
        return None

    # sort by quality according to KLIFS classification
    # high quality structures come first
    # TODO: additional filtering -> Abl1: nilotinib -> 5mo4 (with allosteric ligand and mutations) preferred over 3cs9
    kinases = kinases.sort_values(
        by=["alt", "chain", "quality_score"], ascending=[True, True, False]
    )

    # search for kinase structures with orthosteric ligand
    kinase_complexes = kinases[kinases.ligand != 0]
    if len(kinase_complexes) == 0:  # pick structure with highest quality
        return kinases.iloc[0]
    else:  # pick structure with similar ligand and high quality
        # get resolved structure of orthosteric ligands
        complex_ligands = [
            get_klifs_ligand(structure_id)
            for structure_id in kinase_complexes.structure_ID
        ]

        # get reasonable conformations of ligand of interest
        ligand = read_smiles(smiles)
        tautomers = generate_tautomers(ligand)
        enantiomers = [generate_enantiomers(tautomer) for tautomer in tautomers]
        conformations_ensemble = [
            generate_conformers(enantiomer)
            for enantiomer in itertools.chain.from_iterable(enantiomers)
        ]

        # overlay and score
        scores = []
        for conformations in conformations_ensemble:
            scores += [
                [i, overlay_molecules(complex_ligand, conformations, False)]
                for i, complex_ligand in enumerate(complex_ligands)
            ]
        score_threshold = max([score[1] for score in scores]) - 0.1

        # pick highest quality structure from structures with similar ligands
        index = min([score[0] for score in scores if score[1] >= score_threshold])
        return kinase_complexes.iloc[index]
