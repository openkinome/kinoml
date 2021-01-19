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


def remove_expression_tags(structure):
    """
    Remove expression tags from a protein structure listed in the PDB header section "SEQADV".
    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule with associated PDB header section "SEQADV".
    Returns
    -------
    structure: oechem.OEGraphMol
        The OpenEye molecule without expression tags.
    """
    # retrieve "SEQADV" records from PDB header
    pdb_data_pairs = oechem.OEGetPDBDataPairs(structure)
    seqadv_records = [datapair.GetValue() for datapair in pdb_data_pairs if datapair.GetTag() == "SEQADV"]
    expression_tags = [seqadv_record for seqadv_record in seqadv_records if "EXPRESSION TAG" in seqadv_record]

    # remove expression tags
    for expression_tag in expression_tags:
        chain_id = expression_tag[10]
        residue_name = expression_tag[6:9]
        residue_id = int(expression_tag[12:16])
        hier_view = oechem.OEHierView(structure)
        hier_residue = hier_view.GetResidue(chain_id, residue_name, residue_id)
        for atom in hier_residue.GetAtoms():
            structure.DeleteAtom(atom)
    return structure


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
        Path to OpenEye Spruce loop database. You can request a copy at
        https://www.eyesopen.com/database-downloads. A testing subset (3TPP) is available
        at https://docs.eyesopen.com/toolkits/python/sprucetk/examples_make_design_units.html.
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

    # remove existing OXT atoms, since they prevent proper capping
    predicate = oechem.OEIsHetAtom()
    for atom in structure.GetAtoms():
        if not predicate(atom):
            if atom.GetName().strip() == "OXT":
                structure.DeleteAtom(atom)

    # select primary alternate location
    alt_factory = oechem.OEAltLocationFactory(structure)
    if alt_factory.GetGroupCount() != 0:
        alt_factory.MakePrimaryAltMol(structure)

    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    # alignment options, only matches are important
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(1)
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(-1)
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(0)
    # capping options
    if cap_termini is False:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    # provide path to loop database
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
    from opencadd.databases.klifs import setup_remote

    remote = setup_remote()
    kinase_ids = remote.kinases.all_kinases()["kinase.klifs_id"]
    kinases = remote.kinases.by_kinase_klifs_id(list(kinase_ids))
    kinase = kinases[kinases["kinase.uniprot"] == uniprot_id].iloc[0]
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
    from ..utils import LocalFileStorage

    file_path = LocalFileStorage.klifs_ligand_mol2(structure_id)

    if not file_path.is_file():
        from opencadd.databases.klifs import setup_remote

        remote = setup_remote()
        mol2_text = remote.coordinates.to_text(str(structure_id), entity="ligand", extension="mol2")
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
    molecule: oechem.OEGraphMol,
    max_centers: int = 12,
    force_flip: bool = False,
    enumerate_nitrogens: bool = True,
) -> List[oechem.OEGraphMol]:
    """
    Generate enantiomers of a given molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    max_centers: int
        The maximal number of stereo centers to enumerate.
    force_flip: bool
        If specified stereo centers should be ignored.
    enumerate_nitrogens: bool
        If nitrogens with invertible pyramidal geometry should be enumerated.
    Returns
    -------
    enantiomers: list of oechem.OEGraphMol
        A list of OpenEye molecules holding the enantiomers.
    """
    from openeye import oechem, oeomega

    enantiomers = [
        oechem.OEGraphMol(enantiomer)
        for enantiomer in oeomega.OEFlipper(
            molecule, max_centers, force_flip, enumerate_nitrogens
        )
    ]
    return enantiomers


def generate_conformations(
    molecule: oechem.OEGraphMol, max_conformations: int = 1000, dense: bool = False
) -> oechem.OEMol:
    """
    Generate conformations of a given molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule.
    max_conformations: int
        Maximal number of conformations to generate.
    dense: bool
        If densely sampled conformers should be generated. Will overwrite max_conformations settings.
    Returns
    -------
    conformations: oechem.OEMol
        An OpenEye multi-conformer molecule holding the generated conformations.
    """
    from openeye import oechem, oeomega

    if oeomega.OEIsMacrocycle(molecule):
        omega_options = oeomega.OEMacrocycleOmegaOptions()
        if dense:  # inspired by oeomega.OEOmegaSampling_Dense
            omega_options.SetMaxConfs(20000)
        else:
            omega_options.SetMaxConfs(max_conformations)
        omega = oeomega.OEMacrocycleOmega(omega_options)
    else:
        if dense:
            omega_options = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        else:
            omega_options = oeomega.OEOmegaOptions()
            omega_options.SetMaxSearchTime(60.0)  # time out
            omega_options.SetMaxConfs(max_conformations)
        omega = oeomega.OEOmega(omega_options)
        omega.SetStrictStereo(False)

    conformations = oechem.OEMol(molecule)
    omega.Build(conformations)

    return conformations


def generate_reasonable_conformations(
    molecule: oechem.OEGraphMol, dense: bool = False,
) -> List[oechem.OEMol]:
    """
    Generate conformations of reasonable enantiomers and tautomers of a given molecule.
    Parameters
    ----------
    molecule: oechem.ORGraphMol
        An OpenEye molecule.
    dense: bool
        If densely sampled conformers should be generated.
    Returns
    -------
    conformations_ensemble: list of oechem.OEMol
        A list of multi-conformer molecules.
    """
    import itertools

    tautomers = generate_tautomers(molecule)
    enantiomers = [generate_enantiomers(tautomer) for tautomer in tautomers]
    conformations_ensemble = [
        generate_conformations(enantiomer, dense=dense)
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


def sequence_similarity(
        sequence1: str,
        sequence2: str,
        open_gap_penalty: int = -11,
        extend_gap_penalty: int = -1,
) -> float:
    """
    Compare the characters of two strings.
    Parameters
    ----------
    sequence1: str
        The first sequence.
    sequence2: str
        The second sequence.
    open_gap_penalty: int
        The penalty to open a gap.
    extend_gap_penalty: int
        The penalty to extend a gap.
    Returns
    -------
    score: float
        Similarity of sequences.
    """
    from Bio import pairwise2
    from Bio.Align import substitution_matrices

    blosum62 = substitution_matrices.load("BLOSUM62")
    # replace any characters unknown to the substitution matrix by *
    sequence1_clean = "".join([x if x in blosum62.alphabet else "*" for x in sequence1])
    sequence2_clean = "".join([x if x in blosum62.alphabet else "*" for x in sequence2])
    score = pairwise2.align.globalds(
        sequence1_clean,
        sequence2_clean,
        blosum62,
        open_gap_penalty,
        extend_gap_penalty,
        score_only=True
    )
    return score


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
    import json
    import requests
    import urllib

    ligands = {}
    base_url = "https://data.rcsb.org/graphql?query="
    query = '{chem_comps(comp_ids:[' + \
            ','.join(['"' + ligand_id + '"' for ligand_id in set(ligand_ids)]) + \
            ']){chem_comp{id}rcsb_chem_comp_descriptor{SMILES}}}'
    response = requests.get(base_url + urllib.parse.quote(query))
    for ligand in json.loads(response.text)["data"]["chem_comps"]:
        ligands[ligand["chem_comp"]["id"]] = ligand["rcsb_chem_comp_descriptor"]["SMILES"]
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


def apply_deletions(
    target_structure: oechem.OEGraphMol, template_sequence: str
) -> oechem.OEGraphMol:
    """
    Apply deletions to a protein structure according to an amino acid sequence. The provided protein structure should
    only contain protein residues to prevent unexpected behavior.
    Parameters
    ----------
    target_structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure for which deletions should be applied.
    template_sequence: str
        A template one letter amino acid sequence, which holds potential deletions when compared to the target
        structure sequence.
    Returns
    -------
     : oechem.OEGraphMol
        An OpenEye molecule holding the protein structure with applied deletions.
    """
    from Bio import pairwise2

    target_sequence = get_sequence(target_structure)
    template_sequence_aligned, target_sequence_aligned = pairwise2.align.globalxs(
        template_sequence, target_sequence, -1, 0
    )[0][:2]
    logging.debug(f"Template sequence:\n{template_sequence_aligned}")
    logging.debug(f"Target sequence:\n{target_sequence_aligned}")
    hierview = oechem.OEHierView(target_structure)
    structure_residues = hierview.GetResidues()
    structure_residue = False
    # adjust target structure to match template sequence
    for template_sequence_residue, target_sequence_residue in zip(
            template_sequence_aligned, target_sequence_aligned
    ):
        # iterate over structure residues
        if target_sequence_residue != "-":
            structure_residue = structure_residues.next()
        # delete any residue from target structure not covered by target sequence
        if template_sequence_residue == "-" and structure_residue:
            for atom in structure_residue.GetAtoms():
                target_structure.DeleteAtom(atom)

    return target_structure


def apply_insertions(
    target_structure: oechem.OEGraphMol,
    template_sequence: str,
    loop_db: str,
) -> oechem.OEGraphMol:
    """
    Apply insertions to a protein structure according to an amino acid sequence. The provided protein structure should
    only contain protein residues to prevent unexpected behavior.
    Parameters
    ----------
    target_structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure for which insertions should be applied.
    template_sequence: str
        A template one letter amino acid sequence, which holds potential insertions when compared to the target
        structure sequence.
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    Returns
    -------
     : oechem.OEGraphMol
        An OpenEye molecule holding the protein structure with applied insertions.
    """
    from pathlib import Path

    from Bio import pairwise2

    sidechain_options = oespruce.OESidechainBuilderOptions()
    loop_options = oespruce.OELoopBuilderOptions()
    loop_db = str(Path(loop_db).expanduser().resolve())
    loop_options.SetLoopDBFilename(loop_db)
    # the hierarchy view is more stable if reinitialized after each change
    # https://docs.eyesopen.com/toolkits/python/oechemtk/biopolymers.html#a-hierarchy-view
    finished = False
    while not finished:
        altered = False
        # align template and target sequences
        target_sequence = get_sequence(target_structure)
        template_sequence_aligned, target_sequence_aligned = pairwise2.align.globalxs(
            template_sequence, target_sequence, -1, 0
        )[0][:2]
        logging.debug(f"Template sequence:\n{template_sequence_aligned}")
        logging.debug(f"Target sequence:\n{target_sequence_aligned}")
        hierview = oechem.OEHierView(target_structure)
        structure_residues = hierview.GetResidues()
        gap_sequence = ""
        gap_start = False
        structure_residue = False
        # adjust target structure to match template sequence
        for template_sequence_residue, target_sequence_residue in zip(
                template_sequence_aligned, target_sequence_aligned
        ):
            # check for gap and make sure missing sequence at N terminus is ignored
            if target_sequence_residue == "-" and structure_residue:
                # get last residue before gap and store gap sequence
                gap_start = structure_residue.GetOEResidue()
                gap_sequence += template_sequence_residue
            # iterate over structure residues and check for gap end
            if target_sequence_residue != "-":
                structure_residue = structure_residues.next()
                # existing gap_starts indicates gap end
                if isinstance(gap_start, oechem.OEResidue):
                    # get first residue after gap end
                    gap_end = structure_residue.GetOEResidue()
                    modeled_structure = oechem.OEMol()
                    logging.debug(f"Trying to build loop {gap_sequence} " +
                                  f"between residues {gap_start.GetResidueNumber()}" +
                                  f" and {gap_end.GetResidueNumber()} ...")
                    # build loop and reinitialize if successful
                    if oespruce.OEBuildSingleLoop(
                        modeled_structure,
                        target_structure,
                        gap_sequence,
                        gap_start,
                        gap_end,
                        sidechain_options,
                        loop_options
                    ):
                        # break loop and reinitialize
                        target_structure = oechem.OEGraphMol(modeled_structure)
                        altered = True
                        break
                    # TODO: else, make sure gap_start and gap_end are not connected,
                    #       since this may indicate an isoform specific insertion
        # leave while loop if no changes were introduced
        if not altered:
            finished = True

    return target_structure


def apply_mutations(
    target_structure: oechem.OEGraphMol, template_sequence: str
) -> oechem.OEGraphMol:
    """
    Mutate a protein structure according to an amino acid sequence. The provided protein structure should only contain
    protein residues to prevent unexpected behavior.
    Parameters
    ----------
    target_structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure to mutate.
    template_sequence: str
        A template one letter amino acid sequence, which holds potential mutations when compared to the target
        structure sequence.
    Returns
    -------
     : oechem.OEGraphMol
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
            template_sequence, target_sequence, -1, 0
        )[0][:2]
        logging.debug(f"Template sequence:\n{template_sequence_aligned}")
        logging.debug(f"Target sequence:\n{target_sequence_aligned}")
        hierview = oechem.OEHierView(target_structure)
        structure_residues = hierview.GetResidues()
        # adjust target structure to match template sequence
        for template_sequence_residue, target_sequence_residue in zip(
            template_sequence_aligned, target_sequence_aligned
        ):
            # check for mutations if no gap
            if target_sequence_residue != "-":
                structure_residue = structure_residues.next()
                if template_sequence_residue != "-":
                    if target_sequence_residue != template_sequence_residue:
                        # mutate and reinitialize if successful
                        structure_residue = structure_residue.GetOEResidue()
                        three_letter_code = oechem.OEGetResidueName(
                            oechem.OEGetResidueIndexFromCode(template_sequence_residue)
                        )
                        logging.debug("Trying to perform mutation " +
                                      f"{structure_residue.GetName()}{structure_residue.GetResidueNumber()}" +
                                      f"{three_letter_code} ...")
                        if oespruce.OEMutateResidue(
                            target_structure, structure_residue, three_letter_code
                        ):
                            logging.debug("Success!")
                            # break loop and reinitialize
                            altered = True
                            break
                        else:
                            logging.debug("Fail!")
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


def split_molecule_components(molecule: oechem.OEGraphMol) -> List[oechem.OEGraphMol]:
    """
    Split an OpenEye Molecule into its bonded components.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding multiple components.
    Returns
    -------
    : list of oechem.OEGraphMol
        A list of OpenEye molecules holding the split components.
    """
    # determine bonded components
    number_of_components, part_list = oechem.OEDetermineComponents(molecule)
    predicate = oechem.OEPartPredAtom(part_list)

    # get bonded components
    components = []
    for i in range(number_of_components):
        predicate.SelectPart(i + 1)
        component = oechem.OEGraphMol()
        oechem.OESubsetMol(component, molecule, predicate)
        components.append(component)

    return components


def clashing_heavy_atoms(
        molecule1: oechem.OEGraphMol,
        molecule2: oechem.OEGraphMol,
        cutoff: float = 1.5
) -> bool:
    """
    Evaluates if the heavy atoms of two molecules are clashing.
    Parameters
    ----------
    molecule1: oechem.OEGraphMol
        An OpenEye molecule.
    molecule2: oechem.OEGraphMol
        An OpenEye molecule.
    cutoff: float
        The cutoff that defines an atom clash.
    Returns
    -------
    : bool
        If any atoms of two molecules are clashing.
    """
    from scipy.spatial import cKDTree

    # select heavy atoms
    heavy_atoms1, heavy_atoms2 = oechem.OEGraphMol(), oechem.OEGraphMol()
    oechem.OESubsetMol(heavy_atoms1, molecule1, oechem.OEIsHeavy(), True)
    oechem.OESubsetMol(heavy_atoms2, molecule2, oechem.OEIsHeavy(), True)
    # get coordinates
    coordinates1_list = [
        heavy_atoms1.GetCoords()[x] for x in sorted(heavy_atoms1.GetCoords().keys())
    ]
    coordinates2_list = [
        heavy_atoms2.GetCoords()[x] for x in sorted(heavy_atoms2.GetCoords().keys())
    ]
    # get clashes
    tree1 = cKDTree(coordinates1_list)
    clashes = tree1.query_ball_point(coordinates2_list, cutoff)
    if max([len(i) for i in clashes]) > 0:
        return True

    return False
