import logging
from pathlib import Path
from typing import List, Set, Union, Iterable, Tuple, Dict

from openeye import oechem, oegrid
from scipy.spatial import cKDTree

# TODO: Add space before Parameters and Returns in docstring, check with numpy standard
# TODO: think more about exceptions
# TODO: Think about using openff-toolkit as much as possible and converting only if needed


def read_smiles(smiles: str, add_hydrogens: bool = True) -> oechem.OEGraphMol:
    """
    Read molecule from a smiles string. Explicit hydrogens will be added by default.

    Parameters
    ----------
    smiles: str
        Smiles string.
    add_hydrogens: bool
        If explicit hydrogens should be added.

    Returns
    -------
    molecule: oechem.OEGraphMol
        A molecule as OpenEye molecules.

    Raises
    ------
    ValueError
        Could not interpret input SMILES.
    """
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles)

    molecules = []
    for molecule in ims.GetOEMols():
        if add_hydrogens:
            oechem.OEAddExplicitHydrogens(molecule)
        molecules.append(oechem.OEGraphMol(molecule))

    if len(molecules) == 0:
        raise ValueError("Could not interpret input SMILES.")

    return molecules[0]


def read_molecules(path: Union[str, Path], add_hydrogens: bool = False) -> List[oechem.OEGraphMol]:
    """
    Read molecules from a file. Explicit hydrogens will not be added by default.

    Parameters
    ----------
    path: str, pathlib.Path
        Path to molecule file.
    add_hydrogens: bool
        If explicit hydrogens should be added.

    Returns
    -------
    molecules: list of oechem.OEGraphMol
        A List of molecules as OpenEye molecules.

    Raises
    ------
    ValueError
        Given file does not contain valid molecules.
    """
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
            if add_hydrogens:
                oechem.OEAddExplicitHydrogens(molecule)
            molecules.append(oechem.OEGraphMol(molecule))

    if len(molecules) == 0:
        raise ValueError("Given file does not contain valid molecules.")

    return molecules


def read_electron_density(path: Union[str, Path]) -> Union[oegrid.OESkewGrid, None]:
    """
    Read electron density from a file.

    Parameters
    ----------
    path: str, pathlib.Path
        Path to electron density file.

    Returns
    -------
    electron_density: oegrid.OESkewGrid or None
        A List of molecules as OpenEye molecules.

    Raises
    ------
    ValueError
        Not a valid electron density file or wrong format. Only MTZ is currently supported.
    """
    path = str(Path(path).expanduser().resolve())
    electron_density = oegrid.OESkewGrid()
    # TODO: different map formats
    if not oegrid.OEReadMTZ(path, electron_density, oegrid.OEMTZMapType_Fwt):
        electron_density = None

    if electron_density is None:
        raise ValueError("Not a valid electron density file or wrong format. Only MTZ is currently supported.")

    return electron_density


def write_molecules(molecules: List[oechem.OEGraphMol], path: Union[str, Path]):
    """
    Save molecules to file.

    Parameters
    ----------
    molecules: list of oechem.OEGraphMol
        A list of OpenEye molecules for writing.
    path: str, pathlib.Path
        File path for saving molecules.
    """
    path = str(Path(path).expanduser().resolve())
    with oechem.oemolostream(path) as ofs:
        for molecule in molecules:
            oechem.OEWriteMolecule(ofs, molecule)

    return


def select_chain(molecule: oechem.OEGraphMol, chain_id: str):
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

    Raises
    ------
    ValueError
        No atoms were found with given chain id.
    """
    # do not change input mol
    selection = molecule.CreateCopy()

    # delete other chains
    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if residue.GetChainID() != chain_id:
            selection.DeleteAtom(atom)

    # check if chain was actually present
    if selection.NumAtoms() == 0:
        raise ValueError("No atoms were found with given chain id.")

    return selection


def select_altloc(
        molecule: oechem.OEGraphMol,
        altloc_id: str,
        altloc_fallback: bool = True,
):
    """
    Select an alternate location from an OpenEye molecule.

    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    altloc_id: str
        Alternate location identifier.
    altloc_fallback: bool
        If the alternate location with the highest occupancy should be used for residues that do
        not contain the given alternate location identifier.

    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the selected alternate location.

    Raises
    ------
    ValueError
        No atoms were found with given altloc id.
    """
    # do not change input mol
    selection = molecule.CreateCopy()

    altloc_factory = oechem.OEAltLocationFactory(selection)
    if altloc_fallback:
        altloc_factory.MakePrimaryAltMol(selection)
    altloc_was_present = False
    for atom in altloc_factory.GetAltAtoms():
        if altloc_factory.MakeAltMol(selection, atom, altloc_id):
            altloc_was_present = True

    # check if altloc id was actually present
    if not altloc_was_present:
        raise ValueError("No atoms were found with given altloc id.")

    return selection


def remove_non_protein(
        molecule: oechem.OEGraphMol,
        exceptions: Union[None, List[str]] = None,
        remove_water: bool = False,
) -> oechem.OEGraphMol:
    """
    Remove non-protein atoms from an OpenEye molecule. Water will be kept by default.

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


def delete_residue(
        structure: oechem.OEGraphMol,
        chain_id: str,
        residue_name: str,
        residue_id: int
) -> oechem.OEGraphMol:
    """
    Delete a residue from an OpenEye molecule.

    Parameters
    ---------
    structure: oechem.OEGraphMol
        An OpenEye molecule with residue information.
    chain_id: str
        The chain id of the residue
    residue_name: str
        The residue name in three letter code.
    residue_id: int
        The residue id.

    Returns
    -------
    : oechem.OEGraphMol
        The OpenEye molecule without the residue.

    Raises
    ------
    ValueError
        Defined residue was not found in given structure.
    """
    # do not change input structure
    selection = structure.CreateCopy()

    hier_view = oechem.OEHierView(selection)
    hier_residue = hier_view.GetResidue(chain_id, residue_name, residue_id)

    residue_was_present = False
    for atom in hier_residue.GetAtoms():
        selection.DeleteAtom(atom)
        residue_was_present = True

    if not residue_was_present:
        raise ValueError("Defined residue was not found in given structure.")

    return selection


def get_expression_tags(
        structure: oechem.OEGraphMol,
        labels: Iterable[str] = ("EXPRESSION TAG", "CLONING ARTIFACT"),
) -> List[Dict]:
    """
    Get the chain id, residue name and residue id of residues in expression tags from a protein structure listed in the
    PDB header section "SEQADV".

    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule with associated PDB header section "SEQADV".
    labels: Iterable of str
        The 'SEQADV' labels defining expression tags. Default: ('EXPRESSION TAG', 'CLONING ARTIFACT').

    Returns
    -------
    : list of dict
        The chain id, residue name and residue id of residues in the expression tags.
    """
    # retrieve "SEQADV" records from PDB header
    pdb_data_pairs = oechem.OEGetPDBDataPairs(structure)
    seqadv_records = [datapair.GetValue() for datapair in pdb_data_pairs if datapair.GetTag() == "SEQADV"]
    expression_tag_labels = [
        seqadv_record
        for seqadv_record in seqadv_records
        if any(label in seqadv_record for label in labels)
    ]

    # collect expression tag residue information
    expression_tag_residues = []
    for label in expression_tag_labels:
        expression_tag_residues.append(
            {
                "chain_id": label[10],
                "residue_name": label[6:9],
                "residue_id": int(label[12:16])
            }
        )

    return expression_tag_residues


def assign_caps(
        structure: oechem.OEGraphMol,
        real_termini: Union[Iterable[int] or None] = None
) -> oechem.OEGraphMol:
    """
    Cap N and C termini of the given input structure. Real termini can be protected from capping by providing the
    corresponding residue ids via the 'real_termini' argument.

    Parameters
    ----------
    structure: oechem.OEGraphMol
        The OpenEye molecule holding the protein structure to cap.
    real_termini: iterable of int or None
        The biologically relevant real termini that shpuld be prevented from capping.

    Returns
    -------
    structure: oechem.GraphMol
        The OpenEye molecule holding the capped structure.
    """
    from openeye import oespruce

    def _has_residue_number(atom, residue_numbers=real_termini):
        """Returns True if atom matches any given residue number."""
        residue = oechem.OEAtomGetResidue(atom)
        return any(
            [
                True if residue.GetResidueNumber() == residue_number else False
                for residue_number in residue_numbers
            ]
        )

    # remove existing OXT atoms from non real termini, since they can prevent proper capping
    predicate = oechem.OEIsHetAtom()
    for atom in structure.GetAtoms():
        if not predicate(atom):
            if atom.GetName().strip() == "OXT":
                if real_termini:
                    # exclude real termini from deletion of OXT atoms
                    residue = oechem.OEAtomGetResidue(atom)
                    if not residue.GetResidueNumber() in real_termini:
                        structure.DeleteAtom(atom)
                else:
                    structure.DeleteAtom(atom)

    # cap all termini but biologically real termini
    if real_termini:
        oespruce.OECapTermini(structure, oechem.PyAtomPredicate(_has_residue_number))
    else:
        oespruce.OECapTermini(structure)

    # add hydrogen to newly modeled atoms
    options = oechem.OEPlaceHydrogensOptions()
    options.SetBypassPredicate(oechem.OENotAtom(oespruce.OEIsModeledAtom()))
    oechem.OEPlaceHydrogens(structure, options)

    return structure


def _prepare_structure(
        structure: oechem.OEGraphMol,
        has_ligand: bool = False,
        electron_density: Union[oegrid.OESkewGrid, None] = None,
        loop_db: Union[str, None] = None,
        ligand_name: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
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
    chain_id: str or None
        The chain id of interest. If chain id is None, best chain will be selected according to OESpruce.
    alternate_location: str or None
        The alternate location of interest. If alternate location is None, best alternate location will be selected
        according to OEChem.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.

    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.

    Raises
    ------
    ValueError
        No design unit found with given chain ID, ligand name and alternate location.
    """
    from openeye import oespruce

    def _contains_chain(design_unit, chain_id):
        """Returns True if the design unit contains protein residues with given chain ID."""
        all_components = oechem.OEGraphMol()
        design_unit.GetComponents(all_components, oechem.OEDesignUnitComponents_All)
        hier_view = oechem.OEHierView(all_components)
        for hier_chain in hier_view.GetChains():
            if hier_chain.GetChainID() == chain_id:
                return True
        return False

    def _contains_ligand(design_unit, resname):
        """Returns True if the design unit contains a ligand with given residue name."""
        ligand = oechem.OEGraphMol()
        design_unit.GetLigand(ligand)
        hier_view = oechem.OEHierView(ligand)
        for hier_residue in hier_view.GetResidues():
            if hier_residue.GetResidueName() == resname:
                return True
        return False

    # delete loose protein residues, which make the alignment error prone
    structure = delete_loose_residues(structure)

    # select alternate location
    if alternate_location:
        structure = select_altloc(structure, alternate_location)

    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    # also consider alternate locations outside binding pocket, important for later filtering
    design_unit_options.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(False)
    # alignment options, only matches are important
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(
        oechem.OESeqAlignmentMethod_Identity
    )
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(-1)
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(0)
    # capping options, capping done separately
    design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
    design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    # provide path to loop database
    if loop_db is not None:
        from pathlib import Path

        loop_db = str(Path(loop_db).expanduser().resolve())
        design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

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

    # filter design units for ligand of interest
    if ligand_name is not None:
        logging.debug(f"Filtering design units for ligand with name {ligand_name} ...")
        design_units = [
            design_unit
            for design_unit in design_units
            if _contains_ligand(design_unit, ligand_name)
        ]

    # filter design units for chain of interest
    if chain_id is not None:
        logging.debug(f"Filtering design units for chain with ID {chain_id} ...")
        design_units = [
            design_unit
            for design_unit in design_units
            if _contains_chain(design_unit, chain_id)
        ]

    if len(design_units) == 0:
        raise ValueError("No design unit found with given chain ID, ligand name and alternate location.")
    else:
        design_unit = design_units[0]

    # assign ACE and NME caps except for real termini
    if cap_termini:
        impl = design_unit.GetImpl()
        protein = impl.GetProtein()
        assign_caps(protein, real_termini)

    return design_unit


def prepare_complex(
        protein_ligand_complex: oechem.OEGraphMol,
        electron_density: Union[oegrid.OESkewGrid, None] = None,
        loop_db: Union[str, None] = None,
        ligand_name: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
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
    chain_id: str or None
        The chain id of interest. If chain id is None, best chain will be selected according to OESpruce.
    alternate_location: str or None
        The alternate location of interest. If alternate location is None, best alternate location will be selected
        according to OEChem.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.

    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.

    Raises
    ------
    ValueError
        No design unit found with given chain ID, ligand name and alternate location.
    """
    return _prepare_structure(
        structure=protein_ligand_complex,
        has_ligand=True,
        electron_density=electron_density,
        loop_db=loop_db,
        ligand_name=ligand_name,
        chain_id=chain_id,
        alternate_location=alternate_location,
        cap_termini=cap_termini,
        real_termini=real_termini,
    )


def prepare_protein(
        protein: oechem.OEGraphMol,
        loop_db: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
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
    chain_id: str or None
        The chain id of interest. If chain id is None, best chain will be selected according to OESpruce.
    alternate_location: str or None
        The alternate location of interest. If alternate location is None, best alternate location will be selected
        according to OEChem.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.

    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.

    Raises
    ------
    ValueError
        No design unit found with given chain ID, ligand name and alternate location.
    """
    return _prepare_structure(
        structure=protein,
        loop_db=loop_db,
        chain_id=chain_id,
        alternate_location=alternate_location,
        cap_termini=cap_termini,
        real_termini=real_termini,
    )


def read_klifs_ligand(structure_id: int) -> oechem.OEGraphMol:
    """
    Retrieve and read an orthosteric kinase ligand from KLIFS.

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
        mol2_text = remote.coordinates.to_text(structure_id, entity="ligand", extension="mol2")
        with open(file_path, "w") as wf:
            wf.write(mol2_text)

    molecule = read_molecules(file_path)[0]

    return molecule


def generate_tautomers(
        molecule: Union[oechem.OEMolBase, oechem.OEMCMolBase],
        max_generate: int = 4096,
        max_return: int = 16,
        pKa_norm: bool = True,
) -> List[Union[oechem.OEMolBase, oechem.OEMCMolBase]]:
    """
    Generate reasonable tautomers of a given molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase or oechem.OEMCMolBase
        An OpenEye molecule.
    max_generate: int
        Maximal number of tautomers to generate.
    max_return: int
        Maximal number of tautomers to return.
    pKa_norm: bool
        Assign predominant ionization state at pH ~7.4.

    Returns
    -------
    tautomers: list of oechem.OEMolBase or oechem.OEMCMolBase
        A list of OpenEye molecules holding the tautomers.
    """
    from openeye import oequacpac

    tautomer_options = oequacpac.OETautomerOptions()
    tautomer_options.SetMaxTautomersGenerated(max_generate)
    tautomer_options.SetMaxTautomersToReturn(max_return)
    tautomer_options.SetMaxSearchTime(60)
    tautomer_options.SetRankTautomers(True)
    tautomer_options.SetCarbonHybridization(True)
    tautomer_options.SetMaxZoneSize(50)
    tautomer_options.SetApplyWarts(True)
    tautomers = [
        tautomer for tautomer
        in oequacpac.OEGetReasonableTautomers(
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
    from openeye import oeomega

    enantiomers = [
        oechem.OEGraphMol(enantiomer)
        for enantiomer in oeomega.OEFlipper(
            molecule, max_centers, force_flip, enumerate_nitrogens
        )
    ]
    return enantiomers


def generate_conformations(
        molecule: oechem.OEGraphMol,
        max_conformations: int = 1000,
        dense: bool = False
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
    from openeye import oeomega

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
        molecule: oechem.OEGraphMol,
        dense: bool = False,
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
        molecule1: oechem.OEGraphMol,
        molecule2: oechem.OEGraphMol
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
    residues not perceived as standard amino acid will receive the character 'X'.
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
    for hier_residue in hv.GetResidues():
        residue = hier_residue.GetOEResidue()
        if oechem.OEIsStandardProteinResidue(residue) and not residue.IsHetAtom():
            sequence.append(
                oechem.OEGetAminoAcidCode(
                    oechem.OEGetResidueIndex(residue.GetName().strip())
                )
            )
        else:
            sequence.append("X")
    sequence = "".join(sequence)
    return sequence


def get_structure_sequence_alignment(
        structure: oechem.OEGraphMol,
        sequence: str
) -> Tuple[str, str]:
    """
    Generate an alignment between a protein structure and an amino acid sequence. The provided protein structure should
    only contain protein residues to prevent unexpected behavior. Also, this alignment only works for highly similar
    sequences, i.e. only few mutations, deletions and insertions.
    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure.
    sequence: str
        A one letter amino acid sequence.
    Returns
    -------
     structure_sequence_aligned: str
        The aligned protein structure sequence with gaps denoted as "-".
    sequence_aligned: str
        The aligned amino acid sequence with gaps denoted as "-".
    """
    import re

    from Bio import pairwise2

    def _connected_residues(residue1, residue2):
        """Check if OEResidues are connected."""
        for atom1 in residue1.GetAtoms():
            for atom2 in residue2.GetAtoms():
                if atom1.GetBond(atom2):
                    return True
        return False

    # align template and target sequences
    target_sequence = get_sequence(structure)
    sequence_aligned, structure_sequence_aligned = pairwise2.align.globalxs(
        sequence,
        target_sequence,
        open=-1,
        extend=0
    )[0][:2]

    # correct alignments involving gaps
    hierview = oechem.OEHierView(structure)
    structure_residues = list(hierview.GetResidues())
    gaps = re.finditer("[^-][-]+[^-]", structure_sequence_aligned)
    for gap in gaps:
        gap_start = gap.start() - structure_sequence_aligned[:gap.start() + 1].count("-")
        start_residue = structure_residues[gap_start - 1]
        end_residue = structure_residues[gap_start]
        gap_sequence = sequence_aligned[gap.start():gap.end() - 2]
        # check for connected residues, which could indicate are wrong alignment
        # e.g. ABEDEFG     ABEDEFG
        #      ABE--FG <-> AB--EFG
        if _connected_residues(structure_residues[gap_start],
                               structure_residues[gap_start + 1]):
            # check if gap involves last residue but is connected
            if gap.end() == len(structure_sequence_aligned):
                structure_sequence_aligned = (
                        structure_sequence_aligned[:gap.start() + 1] +
                        gap.group()[1:][::-1] +
                        structure_sequence_aligned[gap.end():]
                )
            else:
                # check two ways to invert gap
                if not _connected_residues(structure_residues[gap_start - 1],
                                           structure_residues[gap_start]):
                    structure_sequence_aligned = (
                            structure_sequence_aligned[:gap.start()] +
                            gap.group()[:-1][::-1] +
                            structure_sequence_aligned[gap.end() - 1:]
                    )
                elif not _connected_residues(structure_residues[gap_start + 1],
                                             structure_residues[gap_start + 2]):
                    structure_sequence_aligned = (
                            structure_sequence_aligned[:gap.start() + 1] +
                            gap.group()[1:][::-1] +
                            structure_sequence_aligned[gap.end():]
                    )
                else:
                    logging.debug(
                        f"Alignment contains insertion with sequence {gap_sequence}" +
                        f" between bonded residues {start_residue.GetResidueNumber()}" +
                        f" and {end_residue.GetResidueNumber()}, " +
                        "keeping original alignment ..."
                    )
                    continue
            logging.debug("Correcting sequence gap ...")

    return structure_sequence_aligned, sequence_aligned


def apply_deletions(
        target_structure: oechem.OEGraphMol,
        template_sequence: str
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
    import re

    # align template and target sequences
    target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
        target_structure, template_sequence)
    logging.debug(f"Template sequence:\n{template_sequence_aligned}")
    logging.debug(f"Target sequence:\n{target_sequence_aligned}")
    hierview = oechem.OEHierView(target_structure)
    structure_residues = list(hierview.GetResidues())
    insertions = re.finditer("^[-]+|[^-]{2}[-]+[^-]{2}|[-]+$", template_sequence_aligned)
    for insertion in insertions:
        insertion_start = insertion.start() - target_sequence_aligned[:insertion.start()].count("-")
        insertion_end = insertion.end() - target_sequence_aligned[:insertion.end()].count("-")
        insertion_residues = structure_residues[insertion_start:insertion_end]
        logging.debug(f"Found insertion! Deleting residues "
                      f"{insertion_residues[0].GetResidueNumber()}-"
                      f"{insertion_residues[-1].GetResidueNumber()} ...")
        # delete atoms
        for insertion_residue in insertion_residues:
            for atom in insertion_residue.GetAtoms():
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
    import re

    from openeye import oespruce

    def _disconnect_residues(protein, residue1, residue2):
        """Break the bond connecting two protein residues."""
        _is_backbone = oechem.OEIsBackboneAtom()
        for atom in residue1.GetAtoms():
            if _is_backbone(atom):
                for bond in atom.GetBonds():
                    bonded_atoms = [bond.GetBgn(), bond.GetEnd()]
                    for bonded_atom in bonded_atoms:
                        bonded_residue = oechem.OEAtomGetResidue(bonded_atom)
                        if bonded_residue.GetResidueNumber() == residue2.GetResidueNumber():
                            logging.debug(
                                "Breaking bond between residues " +
                                f"{residue1.GetResidueNumber()} and " +
                                f"{residue2.GetResidueNumber()}"
                            )
                            protein.DeleteBond(bond)
        return protein

    sidechain_options = oespruce.OESidechainBuilderOptions()
    loop_options = oespruce.OELoopBuilderOptions()
    loop_options.SetOptimizationMaxLoops(5)
    loop_db = str(Path(loop_db).expanduser().resolve())
    loop_options.SetLoopDBFilename(loop_db)
    # the hierarchy view is more stable if reinitialized after each change
    # https://docs.eyesopen.com/toolkits/python/oechemtk/biopolymers.html#a-hierarchy-view
    while True:
        reinitialize = False
        # align template and target sequences
        target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
            target_structure, template_sequence)
        logging.debug(f"Template sequence:\n{template_sequence_aligned}")
        logging.debug(f"Target sequence:\n{target_sequence_aligned}")
        hierview = oechem.OEHierView(target_structure)
        structure_residues = list(hierview.GetResidues())
        gaps = list(re.finditer("[^-][-]+[^-]", target_sequence_aligned))
        gaps = sorted(gaps, key=lambda match: len(match.group()))
        for gap in gaps:
            gap_start = gap.start() - target_sequence_aligned[:gap.start() + 1].count("-")
            start_residue = structure_residues[gap_start]
            end_residue = structure_residues[gap_start + 1]
            gap_sequence = template_sequence_aligned[gap.start() + 1:gap.end() - 1]
            loop_conformations = oechem.OEMol()
            logging.debug(f"Trying to build loop {gap_sequence} " +
                          f"between residues {start_residue.GetResidueNumber()}" +
                          f" and {end_residue.GetResidueNumber()} ...")
            # build loop and reinitialize if successful
            if oespruce.OEBuildSingleLoop(
                    loop_conformations,
                    target_structure,
                    gap_sequence,
                    start_residue.GetOEResidue(),
                    end_residue.GetOEResidue(),
                    sidechain_options,
                    loop_options
            ):
                logging.debug("Successfully built loop conformations!")
                for i, loop_conformation in enumerate(loop_conformations.GetConfs()):
                    # loop modeling from OESpruce can lead to ring penetration, e.g. 3bel
                    # the next step tries to fix those issues
                    logging.debug("Deleting newly modeled side chains with severe clashes ...")
                    loop_conformation = oechem.OEGraphMol(loop_conformation)
                    loop_conformation = delete_clashing_sidechains(loop_conformation)
                    oespruce.OEBuildSidechains(loop_conformation)
                    clashes = len(oespruce.OEGetPartialResidues(loop_conformation))
                    if clashes == 0:
                        # break conformation evaluation
                        target_structure = loop_conformation
                        reinitialize = True
                        break
                    logging.debug(
                        f"Generated loop conformation {i} contains not fixable severe clashes, trying next!"
                    )
            if reinitialize:
                # break and reinitialize
                break
            else:
                # increase number of loops to optimize
                if loop_options.GetOptimizationMaxLoops() == 5:
                    logging.debug("Increasing number of loops to optimize to 25!")
                    loop_options.SetOptimizationMaxLoops(25)
                    # break and reinitialize
                    reinitialize = True
                    break
                else:
                    loop_options.SetOptimizationMaxLoops(5)
                    logging.debug("Failed building loop without clashes, skipping insertion!")
                    # break bond between residues next to insertion
                    # important if an isoform specific insertion failed
                    target_structure = _disconnect_residues(
                        target_structure,
                        start_residue,
                        end_residue
                    )
        # leave while loop
        if not reinitialize:
            break

    # add hydrogen to newly modeled residues
    options = oechem.OEPlaceHydrogensOptions()
    options.SetBypassPredicate(oechem.OENotAtom(oespruce.OEIsModeledAtom()))
    oechem.OEPlaceHydrogens(target_structure, options)

    # order residues and atoms
    oechem.OEPDBOrderAtoms(target_structure)

    return target_structure


def apply_mutations(
        target_structure: oechem.OEGraphMol,
        template_sequence: str
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
    from openeye import oespruce

    # the hierarchy view is more stable if reinitialized after each change
    # https://docs.eyesopen.com/toolkits/python/oechemtk/biopolymers.html#a-hierarchy-view
    while True:
        altered = False
        # align template and target sequences
        target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
            target_structure, template_sequence)
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
                        oeresidue = structure_residue.GetOEResidue()
                        three_letter_code = oechem.OEGetResidueName(
                            oechem.OEGetResidueIndexFromCode(template_sequence_residue)
                        )
                        logging.debug("Trying to perform mutation " +
                                      f"{oeresidue.GetName()}{oeresidue.GetResidueNumber()}" +
                                      f"{three_letter_code} ...")
                        if oespruce.OEMutateResidue(
                                target_structure, oeresidue, three_letter_code
                        ):
                            logging.debug("Successfully mutated residue!")
                            # break loop and reinitialize
                            altered = True
                            break
                        else:
                            logging.debug("Mutation failed! Deleting residue ...")
                            # deleting atoms via structure_residue.GetAtoms()
                            # results in segmentation fault for 2itv
                            for atom in target_structure.GetAtoms():
                                if oechem.OEAtomGetResidue(atom) == oeresidue:
                                    target_structure.DeleteAtom(atom)
        # leave while loop if no changes were introduced
        if not altered:
            break
    # OEMutateResidue doesn't always build side chains
    # and doesn't add hydrogen automatically
    oespruce.OEBuildSidechains(target_structure)
    options = oechem.OEPlaceHydrogensOptions()
    options.SetBypassPredicate(oechem.OENotAtom(oespruce.OEIsModeledAtom()))
    oechem.OEPlaceHydrogens(target_structure, options)
    # update residue information
    oechem.OEPerceiveResidues(target_structure, oechem.OEPreserveResInfo_All)

    return target_structure


def delete_partial_residues(structure: oechem.OEGraphMol) -> oechem.OEGraphMol:
    """
    Delete residues with missing side chain atoms.
    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure.
    Returns
    -------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding only residues with completely modeled side chains.
    """
    from openeye import oespruce

    # try to build missing sidechains
    oespruce.OEBuildSidechains(structure)

    # find residues with missing sidechain atoms
    incomplete_residues = oespruce.OEGetPartialResidues(structure)

    # delete atoms
    for incomplete_residue in incomplete_residues:
        logging.debug(
            "Deleting protein residue with incomplete sidechain "
            f"{incomplete_residue.GetName()}"
            f"{incomplete_residue.GetResidueNumber()}"
        )
        hier_view = oechem.OEHierView(structure)
        structure_residue = hier_view.GetResidue(
            incomplete_residue.GetChainID(),
            incomplete_residue.GetName(),
            incomplete_residue.GetResidueNumber()
        )
        for atom in structure_residue.GetAtoms():
            structure.DeleteAtom(atom)

    # spruce sometimes creates protein residues consisting of water atoms, e.g. 2hz0 chain B
    # spruce does not always delete residues with missing backbone atoms, e.g. 3qrj chain B
    # TODO: submit bug report
    backbone_atom_names = {"C", "CA", "N"}
    hier_view = oechem.OEHierView(structure)
    for hier_residue in hier_view.GetResidues():
        atom_names = set([atom.GetName().strip() for atom in hier_residue.GetAtoms()])
        if len(backbone_atom_names.difference(atom_names)) > 0:
            logging.debug(
                "Deleting protein residue with incomplete backbone "
                f"{hier_residue.GetResidueName()}"
                f"{hier_residue.GetResidueNumber()} ..."
            )
            for atom in hier_residue.GetAtoms():
                structure.DeleteAtom(atom)

    return structure


def delete_short_protein_segments(protein: oechem.OEGraphMol) -> oechem.OEGraphMol:
    """
    Delete protein segments consisting of 3 or less residues. The given molecule should only contain protein residues.

    Parameters
    ----------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a protein with possibly short segments.

    Returns
    -------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding the protein without short segments.
    """
    components = split_molecule_components(protein)
    for component in components:
        residues = set([oechem.OEAtomGetResidue(atom) for atom in component.GetAtoms()])
        if len(residues) <= 3:
            logging.debug(
                "Deleting loose protein segment with resids "
                f"{[residue.GetResidueNumber() for residue in residues]} ..."
            )
            for residue in residues:
                residue_match = oechem.OEAtomMatchResidueID()
                residue_match.SetName(residue.GetName())
                residue_match.SetChainID(residue.GetChainID())
                residue_match.SetResidueNumber(str(residue.GetResidueNumber()))
                residue_predicate = oechem.OEAtomMatchResidue(residue_match)
                for atom in protein.GetAtoms(residue_predicate):
                    protein.DeleteAtom(atom)

    return protein


def delete_loose_residues(structure: oechem.OEGraphMol) -> oechem.OEGraphMol:
    """
    Delete protein residues that are not bonded to any other residue.

    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure with possibly loose residues.

    Returns
    -------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding the protein structure without loose residues.
    """
    # iterate over protein residues
    # defined by C alpha atoms that are not hetero atoms
    for atom in structure.GetAtoms(
            oechem.OEAndAtom(
                oechem.OEIsCAlpha(), oechem.OENotAtom(
                    oechem.OEIsHetAtom()
                )
            )
    ):
        connected_residues = 0
        # get neighboring backbone atoms
        for connected_atom in atom.GetAtoms(
                oechem.OEIsBackboneAtom()
        ):
            # get neighboring backbone carbon or  nitrogen atoms that are not C alpha atoms
            # which will be from the neighboring residues
            for _ in connected_atom.GetAtoms(
                    oechem.OEOrAtom(
                        oechem.OEIsNitrogen(),
                        oechem.OEAndAtom(
                            oechem.OEIsCarbon(),
                            oechem.OENotAtom(
                                oechem.OEIsCAlpha()
                            )
                        )
                    )
            ):
                connected_residues += 1
        # delete residues with less than 1 neighbor
        if connected_residues < 1:
            residue = oechem.OEAtomGetResidue(atom)
            logging.debug(f"Deleting loose residue {residue.GetName()}{residue.GetResidueNumber()}...")
            for residue_atom in structure.GetAtoms(oechem.OEAtomIsInResidue(residue)):
                structure.DeleteAtom(residue_atom)

    return structure


def delete_clashing_sidechains(
        protein: oechem.OEGraphMol,
        cutoff: float = 2.0
) -> oechem.OEGraphMol:
    """
    Delete side chains that are clashing with other atoms of the given protein structure. Structures containing
    non-protein residues may lead to unexpected behavior.

    Parameters
    ----------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure.
    cutoff: float
        The distance cutoff that is used for defining a heavy atom clash.

    Returns
    -------
    : oechem.OEGraphMol
        An OpenEye molecule holding the protein structure without clashing side chains.
    """
    # get all protein heavy atoms and create KD tree for querying
    protein_heavy_atoms = oechem.OEGraphMol()
    oechem.OESubsetMol(
        protein_heavy_atoms,
        protein,
        oechem.OEIsHeavy()
    )
    protein_heavy_atom_coordinates = get_atom_coordinates(protein_heavy_atoms)
    protein_heavy_atom_tree = cKDTree(protein_heavy_atom_coordinates)
    # create a list of side chain components
    # two cysteine side chains connected via a disulfide bond
    # are considered a single side chain component
    sidechain_heavy_atoms = oechem.OEGraphMol()
    oechem.OESubsetMol(
        sidechain_heavy_atoms,
        protein_heavy_atoms,
        oechem.OENotAtom(
            oechem.OEIsBackboneAtom()
        )
    )
    sidechain_components = split_molecule_components(sidechain_heavy_atoms)
    # iterate over side chains and check for clashes
    for sidechain_component in sidechain_components:
        sidechain_coordinates = get_atom_coordinates(sidechain_component)
        sidechain_tree = cKDTree(sidechain_coordinates)
        clashes = protein_heavy_atom_tree.query_ball_tree(sidechain_tree, cutoff)
        residue_ids = set(
            [
                oechem.OEAtomGetResidue(atom).GetResidueNumber()
                for atom in sidechain_component.GetAtoms()
            ]
        )
        proline = 0
        atom_sample = sidechain_component.GetAtoms().next()
        residue_name = oechem.OEAtomGetResidue(atom_sample).GetName().strip()
        if residue_name == "PRO":
            proline = 1
        # check if more atoms are within cutoff than number of side chain atoms
        # plus 1 for each CA atom and plus 1 for proline
        if len([x for x in clashes if len(x) > 0]) > \
                sidechain_component.NumAtoms() + len(residue_ids) + proline:
            logging.debug(
                f"Deleting clashing side chains with residue ids {residue_ids} ..."
            )
            # delete atoms
            for residue_id in residue_ids:
                for atom in protein.GetAtoms(
                        oechem.OEAndAtom(
                            oechem.OEHasResidueNumber(residue_id),
                            oechem.OENotAtom(
                                oechem.OEIsBackboneAtom()
                            )
                        )
                ):
                    protein.DeleteAtom(atom)

    return protein


def get_atom_coordinates(molecule: oechem.OEGraphMol) -> List[Tuple[float, float, float]]:
    """
    Retrieve the atom coordinates of an OpenEye molecule.

    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule for which the coordinates should be retrieved.

    Returns
    -------

    : list of tuple of float
        The coordinates of the given molecule atoms.
    """
    coordinates_dict = molecule.GetCoords()
    # get atom coordinates in order of atom indices
    coordinates = [
        coordinates_dict[key] for key in sorted(coordinates_dict.keys())
    ]
    return coordinates


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
        reference_protein: oechem.OEGraphMol,
        fit_protein: oechem.OEGraphMol,
        residues: Iterable = tuple(),
        chain_id: str = " ",
        insertion_code: str = " "
) -> oechem.OEGraphMol:
    """
    Superpose a protein structure onto a reference protein. The superposition
    can be customized to consider only the specified residues.
    Parameters
    ----------
    reference_protein: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure which will be used as reference during superposition.
    fit_protein: oechem.OEGraphMol
        An OpenEye molecule holding a protein structure which will be superposed onto the reference protein.
    residues: Iterable of str
        Residues that should be used during superposition in format "GLY123".
    chain_id: str
        Chain identifier for residues that should be used during superposition.
    insertion_code: str
        Insertion code for residues that should be used during superposition.
    Returns
    -------
    superposed_protein: oechem.OEGraphMol
        An OpenEye molecule holding the superposed protein structure.
    """
    from openeye import oespruce

    # do not modify input
    superposed_protein = fit_protein.CreateCopy()

    # set superposition method
    options = oespruce.OESuperpositionOptions()
    if len(residues) == 0:
        options.SetSuperpositionType(oespruce.OESuperpositionType_Global)
    else:
        options.SetSuperpositionType(oespruce.OESuperpositionType_Site)
        for residue in residues:
            options.AddSiteResidue(f"{residue[:3]}:{residue[3:]}:{insertion_code}:{chain_id}")

    # perform superposition
    superposition = oespruce.OEStructuralSuperposition(
        reference_protein, superposed_protein, options
    )
    superposition.Transform(superposed_protein)

    return superposed_protein


def update_residue_identifiers(
        structure: oechem.OEGraphMol,
        keep_protein_residue_ids: bool = True
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
    # update residue identifiers, except for residue number,
    # residue names, atom names, chain id, record type and insert code
    preserved_info = (
            oechem.OEPreserveResInfo_ResidueNumber
            | oechem.OEPreserveResInfo_ResidueName
            | oechem.OEPreserveResInfo_AtomName
            | oechem.OEPreserveResInfo_ChainID
            | oechem.OEPreserveResInfo_HetAtom
            | oechem.OEPreserveResInfo_InsertCode
    )
    oechem.OEPerceiveResidues(structure, preserved_info)

    # update protein
    residue_number = 0
    hierarchical_view = oechem.OEHierView(structure)
    for hv_residue in hierarchical_view.GetResidues():
        residue = hv_residue.GetOEResidue()
        if not residue.IsHetAtom():
            if keep_protein_residue_ids:
                if residue.GetName() == "NME":
                    # NME residues may have same id as preceding residue
                    residue_number += 1
                else:
                    # catch protein residue id if those should not be touched
                    residue_number = residue.GetResidueNumber()
            else:
                residue_number += 1
            # apply changes to each atom
            for atom in hv_residue.GetAtoms():
                residue = oechem.OEAtomGetResidue(atom)
                residue.SetChainID("A")
                residue.SetResidueNumber(residue_number)
                oechem.OEAtomSetResidue(atom, residue)

    # update all hetero atoms but water
    for hv_residue in hierarchical_view.GetResidues():
        residue = hv_residue.GetOEResidue()
        if residue.IsHetAtom() and residue.GetName().strip() != "HOH":
            residue_number += 1
            # apply changes to each atom
            for atom in hv_residue.GetAtoms():
                residue = oechem.OEAtomGetResidue(atom)
                residue.SetChainID("A")
                residue.SetResidueNumber(residue_number)
                oechem.OEAtomSetResidue(atom, residue)

    # update water
    for hv_residue in hierarchical_view.GetResidues():
        residue = hv_residue.GetOEResidue()
        if residue.IsHetAtom() and residue.GetName().strip() == "HOH":
            residue_number += 1
            # apply changes to each atom
            for atom in hv_residue.GetAtoms():
                residue = oechem.OEAtomGetResidue(atom)
                residue.SetChainID("A")
                residue.SetResidueNumber(residue_number)
                oechem.OEAtomSetResidue(atom, residue)

    # order atoms into PDB order
    oechem.OEPDBOrderAtoms(structure)

    # update residue identifiers, except for residue number,
    # residue names, atom names, chain id and record type
    preserved_info = (
            oechem.OEPreserveResInfo_ResidueNumber
            | oechem.OEPreserveResInfo_ResidueName
            | oechem.OEPreserveResInfo_AtomName
            | oechem.OEPreserveResInfo_ChainID
            | oechem.OEPreserveResInfo_HetAtom
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
