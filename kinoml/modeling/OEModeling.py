import logging
from pathlib import Path
from typing import List, Set, Union, Iterable, Tuple, Dict

from openeye import oechem, oegrid, oeomega
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


def read_electron_density(path: Union[str, Path]) -> oegrid.OESkewGrid:
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


def write_molecules(molecules: List[oechem.OEMolBase], path: Union[str, Path]):
    """
    Save molecules to file.

    Parameters
    ----------
    molecules: list of oechem.OEMolBase
        A list of OpenEye molecules for writing.
    path: str, pathlib.Path
        File path for saving molecules.
    """
    path = str(Path(path).expanduser().resolve())
    with oechem.oemolostream(path) as ofs:
        for molecule in molecules:
            oechem.OEWriteMolecule(ofs, molecule)

    return


def select_chain(molecule: oechem.OEMolBase, chain_id: str) -> oechem.OEMolBase:
    """
    Select a chain from an OpenEye molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule holding a molecular structure.
    chain_id: str
        Chain identifier.

    Returns
    -------
    selection: oechem.OEMolBase
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
        molecule: oechem.OEMolBase,
        altloc_id: str,
        altloc_fallback: bool = True,
) -> oechem.OEMolBase:
    """
    Select an alternate location from an OpenEye molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule holding a molecular structure.
    altloc_id: str
        Alternate location identifier.
    altloc_fallback: bool
        If the alternate location with the highest occupancy should be used for residues that do
        not contain the given alternate location identifier.

    Returns
    -------
    selection: oechem.OEMolBase
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

    # remove alternate location identifiers
    oechem.OEPerceiveResidues(
        selection,
        oechem.OEPreserveResInfo_All - oechem.OEPreserveResInfo_AlternateLocation
    )

    return selection


def remove_non_protein(
        molecule: oechem.OEMolBase,
        exceptions: Union[None, List[str]] = None,
        remove_water: bool = False,
) -> oechem.OEMolBase:
    """
    Remove non-protein atoms from an OpenEye molecule. Water will be kept by default.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule holding a molecular structure.
    exceptions: None or list of str
        Exceptions that should not be removed.
    remove_water: bool
        If water should be removed.

    Returns
    -------
    selection: oechem.OEMolBase
        An OpenEye molecule holding the filtered structure.
    """
    non_standards_amino_acids = [  # Nagata 2014 (10.1093/bioinformatics/btu106)
        "ABA", "CSO", "CSD", "CME", "OCS", "KCX", "LLP", "MLY", "M3L", "MSE", "PCA", "HYP", "SEP",
        "TPO", "PTR"
    ]
    if exceptions is None:
        exceptions = []
    if remove_water is False:
        exceptions.append("HOH")
    exceptions += non_standards_amino_acids

    # do not change input mol
    selection = molecule.CreateCopy()

    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if residue.IsHetAtom():
            if residue.GetName() not in exceptions:
                selection.DeleteAtom(atom)

    return selection


def delete_residue(
        structure: oechem.OEMolBase,
        chain_id: str,
        residue_name: str,
        residue_id: int
) -> oechem.OEGraphMol:
    """
    Delete a residue from an OpenEye molecule.

    Parameters
    ---------
    structure: oechem.OEMolBase
        An OpenEye molecule with residue information.
    chain_id: str
        The chain id of the residue
    residue_name: str
        The residue name in three letter code.
    residue_id: int
        The residue id.

    Returns
    -------
    : oechem.OEMolBase
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
        structure: oechem.OEMolBase,
        labels: Iterable[str] = ("EXPRESSION TAG", "CLONING ARTIFACT"),
) -> List[Dict]:
    """
    Get the chain id, residue name and residue id of residues in expression tags from a protein structure listed in the
    PDB header section "SEQADV".

    Parameters
    ----------
    structure: oechem.OEMolBase
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
        structure: oechem.OEMolBase,
        real_termini: Union[Iterable[int] or None] = None
) -> oechem.OEMolBase:
    """
    Cap N and C termini of the given input structure. Real termini can be protected from capping
    by providing the corresponding residue ids via the 'real_termini' argument.

    Parameters
    ----------
    structure: oechem.OEMolBase
        The OpenEye molecule holding the protein structure to cap.
    real_termini: iterable of int or None
        The biologically relevant real termini that should be prevented from capping.

    Returns
    -------
    structure: oechem.OEMolBase
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

    # fix backbone, i.e. add missing OXT atoms
    oechem.OEClearPDBData(structure)  # prevent modeling based on PDB header
    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)  # no capping
    design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)  # no capping
    design_unit_options.GetPrepOptions().SetProtonate(False)  # add hydrogens later
    design_unit = list(
        oespruce.OEMakeBioDesignUnits(structure, structure_metadata, design_unit_options)
    )[0]
    oespruce.OEFixBackbone(design_unit)  # fix backbone (only available for design units)
    design_unit.GetComponents(structure, oechem.OEDesignUnitComponents_All)

    # add hydrogens to newly modeled atoms
    place_hydrogens_options = oechem.OEPlaceHydrogensOptions()
    place_hydrogens_options.SetBypassPredicate(oechem.OENotAtom(oespruce.OEIsModeledAtom()))
    oechem.OEPlaceHydrogens(structure, place_hydrogens_options)

    # delete lonely hydrogens, e.g. 4f8o
    for atom in structure.GetAtoms():
        if atom.GetDegree() == 0:
            if atom.GetAtomicNum() == 1:
                structure.DeleteAtom(atom)

    return structure


def prepare_structure(
        structure: oechem.OEMolBase,
        has_ligand: bool = False,
        electron_density: Union[oegrid.OESkewGrid, None] = None,
        loop_db: Union[str, None] = None,
        ligand_name: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
        cap_termini: bool = True,
        real_termini: Union[List[int], None] = None,
) -> oechem.OEDesignUnit:
    """
    Prepare an OpenEye molecule holding a protein ligand complex for docking.

    Parameters
    ----------
    structure: oechem.OEMolBase
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
    design_unit: oechem.OEDesignUnit
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
        protein = oechem.OEGraphMol()
        design_unit.GetProtein(protein)
        hier_view = oechem.OEHierView(protein)
        for hier_chain in hier_view.GetChains():
            if hier_chain.GetChainID() == chain_id:
                return True
        return False

    def _update_ligand(design_unit, resname, chain_id):
        """Update ligand of the design unit."""
        chain_ids = []
        if not chain_id:  # get chain ID(s) from protein
            protein = oechem.OEGraphMol()
            design_unit.GetProtein(protein)
            hier_view = oechem.OEHierView(protein)
            for hier_chain in hier_view.GetChains():
                chain_ids.append(hier_chain.GetChainID())
        else:
            chain_ids.append(chain_id)

        components = oechem.OEGraphMol()
        design_unit.GetComponents(components, oechem.OEDesignUnitComponents_All)
        components = split_molecule_components(components)
        for component in components:
            residue = oechem.OEAtomGetResidue(component.GetAtoms().next())
            if residue.GetName() == resname:
                if residue.GetChainID() in chain_ids:
                    oechem.OEUpdateDesignUnit(
                        design_unit, component, oechem.OEDesignUnitComponents_Ligand
                    )
                    return True

        return False

    def _contains_ligand(design_unit, resname, chain_id):
        """
        Returns True if the design unit contains a ligand with given residue name and chain ID.
        """
        ligand = oechem.OEGraphMol()
        design_unit.GetLigand(ligand)
        hier_view = oechem.OEHierView(ligand)
        for hier_residue in hier_view.GetResidues():
            if hier_residue.GetResidueName() == resname:
                if chain_id:
                    if hier_residue.GetOEResidue().GetChainID() == chain_id:
                        return True
                else:
                    return True

        if _update_ligand(design_unit, resname, chain_id):  # e.g. ANP of 3sls
            return True
        
        return False

    # delete short protein segments, which make the alignment error prone
    structure = delete_short_protein_segments(structure)

    # select alternate location
    if alternate_location:
        structure = select_altloc(structure, alternate_location)

    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    # turn off superposition
    design_unit_options.SetSuperpose(False)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment
    design_unit_options.GetSplitOptions().SetMinLigAtoms(5)
    # also consider alternate locations outside binding pocket, important for later filtering
    design_unit_options.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(False)
    # alignment options, only matches are important
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(
        oechem.OESeqAlignmentMethod_Identity
    )
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(-1)
    design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(0)
    # capping options, capping done separately if `real_termini` given
    if not cap_termini or real_termini:
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
        logging.debug(
            f"Filtering design units for ligand with name {ligand_name} and chain ID {chain_id}..."
        )
        design_units = [
            design_unit
            for design_unit in design_units
            if _contains_ligand(design_unit, ligand_name, chain_id)
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
    if cap_termini and real_termini:
        impl = design_unit.GetImpl()
        protein = impl.GetProtein()
        assign_caps(protein, real_termini)

    return design_unit


def prepare_complex(
        protein_ligand_complex: oechem.OEMolBase,
        electron_density: Union[oegrid.OESkewGrid, None] = None,
        loop_db: Union[str, None] = None,
        ligand_name: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
        cap_termini: bool = True,
        real_termini: Union[List[int], None] = None,
) -> oechem.OEDesignUnit:
    """
    Prepare an OpenEye molecule holding a protein ligand complex for docking.

    Parameters
    ----------
    protein_ligand_complex: oechem.OEMolBase
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
    design_unit: oechem.OEDesignUnit
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.

    Raises
    ------
    ValueError
        No design unit found with given chain ID, ligand name and alternate location.
    """
    return prepare_structure(
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
        protein: oechem.OEMolBase,
        loop_db: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
        cap_termini: bool = True,
        real_termini: Union[List[int], None] = None,
) -> oechem.OEDesignUnit:
    """
    Prepare an OpenEye molecule holding a protein structure for docking.

    Parameters
    ----------
    protein: oechem.OEMolBase
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
    return prepare_structure(
        structure=protein,
        loop_db=loop_db,
        chain_id=chain_id,
        alternate_location=alternate_location,
        cap_termini=cap_termini,
        real_termini=real_termini,
    )


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
        Assign the predominant ionization state at pH ~7.4.

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
    ]  # ToDo: report unexpected behavior, with pKa_norm set False, positively charged imidazole become neutral
    return tautomers


def generate_enantiomers(
        molecule: oechem.OEMolBase,
        max_centers: int = 12,
        force_flip: bool = False,
        enumerate_nitrogens: bool = True,
) -> List[oechem.OEMolBase]:
    """
    Generate enantiomers of a given molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule.
    max_centers: int
        The maximal number of stereo centers to enumerate.
    force_flip: bool
        If specified stereo centers should be enumerated.
    enumerate_nitrogens: bool
        If nitrogens with invertible pyramidal geometry should be enumerated.

    Returns
    -------
    enantiomers: list of oechem.OEMolBase
        A list of OpenEye molecules holding the enantiomers.
    """
    flipper_options = oeomega.OEFlipperOptions()
    flipper_options.SetMaxCenters(max_centers)
    flipper_options.SetEnumSpecifiedStereo(force_flip)
    flipper_options.SetEnumNitrogen(enumerate_nitrogens)
    flipper_options.SetWarts(True)

    enantiomers = [
        enantiomer for enantiomer
        in oeomega.OEFlipper(molecule, flipper_options)
    ]
    return enantiomers


def generate_conformations(
        molecule: oechem.OEMolBase,
        options: oeomega.OEOmegaOptions = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Classic)
) -> oechem.OEMCMolBase:
    """
    Generate conformations of a given molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule.
    options: oeomega.OEOmegaOptions, default=oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Classic)
        Options for generating conformations. If the given molecule is a macrocycle only the
        maximal number of conformations will be changed from the defaults defined in
        `oeomega.OEMacrocycleOmegaOptions()`.

    Returns
    -------
    conformations: oechem.OEMCMolBase
        An OpenEye multi-conformer molecule holding the generated conformations.
    """
    if oeomega.OEIsMacrocycle(molecule):
        omega_options = oeomega.OEMacrocycleOmegaOptions()
        # check if range is specified, e.g. via oeomega.OEOmegaSampling_Pose
        conformation_range = options.GetMaxConfRange()
        if len(conformation_range) > 0:
            omega_options.SetMaxConfs(conformation_range[-1])
        else:
            omega_options.SetMaxConfs(options.GetMaxConfs())
        omega = oeomega.OEMacrocycleOmega(omega_options)
    else:
        omega = oeomega.OEOmega(options)

    conformations = oechem.OEMol(molecule)
    omega.Build(conformations)

    return conformations


def generate_reasonable_conformations(
        molecule: oechem.OEMolBase,
        options: oeomega.OEOmegaOptions = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Classic),
        pKa_norm: bool = True,
) -> List[oechem.OEMCMolBase]:
    """
    Generate conformations of reasonable enantiomers and tautomers of a given molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule.
    options: oeomega.OEOmegaOptions, default=oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Classic)
        Options for generating conformations. If the given molecule is a macrocycle only the
        maximal number of conformations will be changed from the defaults defined in
        `oeomega.OEMacrocycleOmegaOptions()`.
    pKa_norm: bool
        Assign the predominant ionization state at pH ~7.4.

    Returns
    -------
    conformations_ensemble: list of oechem.OEMCMolBase
        A list of OpenEye multi-conformer molecules.
    """
    import itertools

    tautomers = generate_tautomers(molecule, pKa_norm=pKa_norm)
    enantiomers = [generate_enantiomers(tautomer) for tautomer in tautomers]
    conformations_ensemble = [
        generate_conformations(enantiomer, options)
        for enantiomer in itertools.chain.from_iterable(enantiomers)
    ]
    return conformations_ensemble


def overlay_molecules(
        reference_molecule: oechem.OEMolBase,
        fit_molecule: oechem.OEMCMolBase,
) -> (float, List[oechem.OEGraphMol]):
    """
    Overlay a multi-conformer molecule to a single-conformer molecule and calculate the TanimotoCombo score.

    Parameters
    ----------
    reference_molecule: oechem.OEMolBase
        An OpenEye molecule holding a single conformation of the reference molecule for overlay.
    fit_molecule: oechem.OEMCMolBase
        An OpenEye multi-conformer molecule holding the conformations of a molecule to fit during overlay.

    Returns
    -------
    : float, list of oechem.OEGraphMol
        The TanimotoCombo score and the OpenEye molecules of the best overlay
    """
    from openeye import oeshape

    # prepare molecules for overlay
    prep = oeshape.OEOverlapPrep()
    prep.Prep(reference_molecule)
    prep.Prep(fit_molecule)

    # perform overlay
    overlay = oeshape.OEOverlay()
    overlay.SetupRef(reference_molecule)
    score = oeshape.OEBestOverlayScore()
    overlay.BestOverlay(score, fit_molecule, oeshape.OEHighestTanimoto())

    # collect results
    best_overlay = [reference_molecule]
    fit_molecule = oechem.OEGraphMol(
        fit_molecule.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx()))
    )
    score.Transform(fit_molecule)
    best_overlay.append(fit_molecule)

    return score.GetTanimotoCombo(), best_overlay


def enumerate_isomeric_smiles(molecule: oechem.OEMolBase) -> Set[str]:
    """
    Enumerate reasonable isomeric SMILES representations of a given OpenEye molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule.

    Returns
    -------
    smiles_set: set of str
        A set of reasonable isomeric SMILES strings.
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


def are_identical_molecules(
        molecule1: oechem.OEMolBase,
        molecule2: oechem.OEMolBase
) -> bool:
    """
    Check if two OpenEye molecules are identical.

    Parameters
    ----------
    molecule1: oechem.OEMolBase
        The first OpenEye molecule.
    molecule2: oechem.OEMolBase
        The second OpenEye molecule.

    Returns
    -------
    : bool
        True if identical molecules, else False.
    """
    isomeric_smiles_set1 = enumerate_isomeric_smiles(molecule1)
    isomeric_smiles_set2 = enumerate_isomeric_smiles(molecule2)

    if len(isomeric_smiles_set1 & isomeric_smiles_set2) == 0:
        return False
    else:
        return True


def get_sequence(structure: oechem.OEMolBase) -> str:
    """
    Get the amino acid sequence with one letter characters of an OpenEye molecule.
    All residues not perceived as standard amino acid will receive the character 'X'.

    Parameters
    ----------
    structure: oechem.OEMolBase
        An OpenEye molecule.

    Returns
    -------
    sequence: str
        The amino acid sequence with one letter characters.
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
        structure: oechem.OEMolBase,
        sequence: str
) -> Tuple[str, str]:
    """
    Generate an alignment between a protein structure and an amino acid sequence. The provided protein structure should
    only contain protein residues to prevent unexpected behavior. Also, this alignment was optimized for highly similar
    sequences, i.e. only few mutations, deletions and insertions. Non protein residues will be marked with "X".

    Parameters
    ----------
    structure: oechem.OEMolBase
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
        # check for connected residues, which could indicate a wrong alignment
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
                    # i.e. ABEDEFG     ABEDEFG
                    #      ABE--FG --> AB--EFG
                    structure_sequence_aligned = (
                            structure_sequence_aligned[:gap.start()] +
                            gap.group()[:-1][::-1] +
                            structure_sequence_aligned[gap.end() - 1:]
                    )
                elif not _connected_residues(structure_residues[gap_start + 1],
                                             structure_residues[gap_start + 2]):
                    # i.e. ABEDEFG     ABEDEFG
                    #      AB--EFG --> AB--EFG
                    structure_sequence_aligned = (
                            structure_sequence_aligned[:gap.start() + 1] +
                            gap.group()[1:][::-1] +
                            structure_sequence_aligned[gap.end():]
                    )
                else:
                    # i.e. ABEDEFG     ABEDEFG
                    #      AB**EFG --> AB--EFG
                    logging.debug(
                        f"Alignment contains insertion with sequence {gap_sequence}" +
                        f" between bonded residues {start_residue.GetResidueNumber()}" +
                        f" and {end_residue.GetResidueNumber()}, " +
                        "keeping original alignment ..."
                    )
                    continue
            logging.debug("Corrected sequence gap ...")

    return structure_sequence_aligned, sequence_aligned


def apply_deletions(
        target_structure: oechem.OEMolBase,
        template_sequence: str,
        delete_n_anchors: int = 2,
) -> oechem.OEMolBase:
    """
    Apply deletions to a protein structure according to an amino acid sequence. The provided protein structure should
    only contain protein residues to prevent unexpected behavior.

    Parameters
    ----------
    target_structure: oechem.OEMolBase
        An OpenEye molecule holding a protein structure for which deletions should be applied.
    template_sequence: str
        A template one letter amino acid sequence, which holds potential deletions when compared to the target
        structure sequence.
    delete_n_anchors: int
        Specify how many anchoring residues should be deleted at each side of the deletion. Important if connecting
        anchoring residues after deletion is intended, e.g. via apply_insertion. Only affects deletions in the middle
        of a sequence, not at the end or the beginning.

    Returns
    -------
    structure_with_deletions: oechem.OEMolBase
        An OpenEye molecule holding the protein structure with applied deletions.

    Raises
    ------
    ValueError
        Negative values are not allowed for 'delete_n_anchors'.
    """
    import re

    if delete_n_anchors < 0:
        raise ValueError("Negative values are not allowed for 'delete_n_anchors'.")

    # do not change input structure
    structure_with_deletions = target_structure.CreateCopy()

    # align template and target sequences
    target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
        structure_with_deletions, template_sequence
    )
    logging.debug(f"Template sequence:\n{template_sequence_aligned}")
    logging.debug(f"Target sequence:\n{target_sequence_aligned}")
    hierview = oechem.OEHierView(structure_with_deletions)
    structure_residues = list(hierview.GetResidues())
    insertions = re.finditer(
        "^[-]+|[^-]{" + str(delete_n_anchors) + "}[-]+[^-]{" + str(delete_n_anchors) + "}|[-]+$",
        template_sequence_aligned
    )
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
                structure_with_deletions.DeleteAtom(atom)

    return structure_with_deletions


def apply_insertions(
        target_structure: oechem.OEMolBase,
        template_sequence: str,
        loop_db: Union[str, Path],
        ligand: Union[oechem.OEMolBase, None] = None,
) -> oechem.OEMolBase:
    """
    Apply insertions to a protein structure according to an amino acid sequence. The provided protein structure should
    only contain protein residues to prevent unexpected behavior.

    Parameters
    ----------
    target_structure: oechem.OEMolBase
        An OpenEye molecule holding a protein structure for which insertions should be applied.
    template_sequence: str
        A template one letter amino acid sequence, which holds potential insertions when compared to the target
        structure sequence.
    loop_db: str or Path
        The path to the loop database used by OESpruce to model missing loops.
    ligand: oechem.OEMolBase or None, default=None
        An OpenEye molecule that should be checked for heavy atom clashes with built insertions.

    Returns
    -------
    structure_with_insertions: oechem.OEMolBase
        An OpenEye molecule holding the protein structure with applied insertions.
    """
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

    # do not change input structures
    structure_with_insertions = target_structure.CreateCopy()
    if ligand is not None:
        ligand_heavy_atoms = ligand.CreateCopy()
        oechem.OESuppressHydrogens(ligand_heavy_atoms)

    sidechain_options = oespruce.OESidechainBuilderOptions()
    loop_options = oespruce.OELoopBuilderOptions()
    loop_options.SetOptimizationMaxLoops(25)
    loop_db = str(Path(loop_db).expanduser().resolve())
    loop_options.SetLoopDBFilename(loop_db)
    # the hierarchy view is more stable if reinitialized after each change
    # https://docs.eyesopen.com/toolkits/python/oechemtk/biopolymers.html#a-hierarchy-view
    while True:
        reinitialize = False
        # align template and target sequences
        target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
            structure_with_insertions, template_sequence)
        logging.debug(f"Template sequence:\n{template_sequence_aligned}")
        logging.debug(f"Target sequence:\n{target_sequence_aligned}")
        hierview = oechem.OEHierView(structure_with_insertions)
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
                    structure_with_insertions,
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
                    loop_conformation = oechem.OEGraphMol(loop_conformation)
                    loop_conformation = delete_clashing_sidechains(loop_conformation)
                    oespruce.OEBuildSidechains(loop_conformation)
                    clashes = len(oespruce.OEGetPartialResidues(loop_conformation))
                    if ligand is not None:  # check for clashes with ligand
                        loop_conformation_heavy_atoms = loop_conformation.CreateCopy()
                        oechem.OESuppressHydrogens(loop_conformation_heavy_atoms)
                        clashes += len(list(
                            oechem.OEGetNearestNbrs(
                                loop_conformation_heavy_atoms,
                                ligand_heavy_atoms, 2
                            )
                        ))
                    if clashes == 0:
                        # break conformation evaluation
                        structure_with_insertions = loop_conformation
                        reinitialize = True
                        break
                    logging.debug(
                        f"Generated loop conformation {i} contains not fixable severe clashes, trying next!"
                    )
            if reinitialize:
                # break and reinitialize
                break
            else:
                logging.debug("Failed building loop without clashes, skipping insertion!")
                # break bond between residues next to insertion
                # important if an isoform specific insertion failed
                structure_with_insertions = _disconnect_residues(
                    structure_with_insertions,
                    start_residue,
                    end_residue
                )
        # leave while loop
        if not reinitialize:
            break

    # add hydrogen to newly modeled residues
    options = oechem.OEPlaceHydrogensOptions()
    options.SetBypassPredicate(oechem.OENotAtom(oespruce.OEIsModeledAtom()))
    oechem.OEPlaceHydrogens(structure_with_insertions, options)

    # order residues and atoms
    oechem.OEPDBOrderAtoms(structure_with_insertions)

    return structure_with_insertions


def apply_mutations(
        target_structure: oechem.OEMolBase,
        template_sequence: str,
        fallback_delete: bool = True,
) -> oechem.OEMolBase:
    """
    Mutate a protein structure according to an amino acid sequence. The provided protein structure should only contain
    protein residues to prevent unexpected behavior. Residues that could not be mutated will be deleted by default.

    Parameters
    ----------
    target_structure: oechem.OEMolBase
        An OpenEye molecule holding a protein structure to mutate.
    template_sequence: str
        A template one letter amino acid sequence, which holds potential mutations when compared to the target
        structure sequence.
    fallback_delete: bool
        If the residue should be deleted if it could not be mutated.

    Returns
    -------
     : oechem.OEMolBase
        An OpenEye molecule holding the mutated protein structure.

    Raises
    ------
    ValueError
        Mutation {oeresidue.GetName()}{oeresidue.GetResidueNumber()}{three_letter_code} failed!
        Only raised when fallback_delete is set False.
    """
    from openeye import oespruce

    # do not change input structure
    structure_with_mutations = target_structure.CreateCopy()

    # the hierarchy view is more stable if reinitialized after each change
    # https://docs.eyesopen.com/toolkits/python/oechemtk/biopolymers.html#a-hierarchy-view
    while True:
        altered = False
        # align template and target sequences
        target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
            structure_with_mutations, template_sequence)
        logging.debug(f"Template sequence:\n{template_sequence_aligned}")
        logging.debug(f"Target sequence:\n{target_sequence_aligned}")
        hierview = oechem.OEHierView(structure_with_mutations)
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
                                structure_with_mutations, oeresidue, three_letter_code
                        ):
                            logging.debug("Successfully mutated residue!")
                            # break loop and reinitialize
                            altered = True
                            break
                        else:
                            if fallback_delete:
                                logging.debug("Mutation failed! Deleting residue ...")
                                atom_functor = oechem.OEOrAtom(
                                    # if residue was not mutated
                                    oechem.OEAtomMatchResidue([
                                        f"{oeresidue.GetName()}:{oeresidue.GetResidueNumber()}"
                                        f":.*:{oeresidue.GetChainID()}:.*"
                                    ]),
                                    # if residue was mutated but side chain chopped of
                                    oechem.OEAtomMatchResidue([
                                        f"{three_letter_code}:{oeresidue.GetResidueNumber()}"
                                        f":.*:{oeresidue.GetChainID()}:.*"
                                    ])
                                )
                                for atom in structure_with_mutations.GetAtoms(atom_functor):
                                    structure_with_mutations.DeleteAtom(atom)
                                # break loop and reinitialize
                                altered = True
                                break
                            else:
                                raise ValueError(
                                    f"Mutation {oeresidue.GetName()}" +
                                    f"{oeresidue.GetResidueNumber()}" +
                                    f"{three_letter_code} failed!"
                                )
        # leave while loop if no changes were introduced
        if not altered:
            break
    # OEMutateResidue doesn't always build side chains
    # and doesn't add hydrogen automatically
    oespruce.OEBuildSidechains(structure_with_mutations)
    options = oechem.OEPlaceHydrogensOptions()
    options.SetBypassPredicate(oechem.OENotAtom(oespruce.OEIsModeledAtom()))
    oechem.OEPlaceHydrogens(structure_with_mutations, options)
    # update residue information
    oechem.OEPerceiveResidues(structure_with_mutations, oechem.OEPreserveResInfo_All)

    return structure_with_mutations


def delete_partial_residues(
        structure: oechem.OEMolBase,
) -> oechem.OEMolBase:
    """
    Delete residues with missing sidechain or backbone atoms. The backbone is considered complete
    if atoms C, CA and N are present.

    Parameters
    ----------
    structure: oechem.OEMolBase
        An OpenEye molecule holding a protein structure.

    Returns
    -------
    structure: oechem.OEMolBase
        An OpenEye molecule holding only residues with completely modeled side chains.
    """
    from openeye import oespruce

    # do not change input structure
    processed_structure = structure.CreateCopy()

    # try to build missing sidechains
    oespruce.OEBuildSidechains(structure)

    # find residues with missing sidechain atoms
    incomplete_residues = oespruce.OEGetPartialResidues(processed_structure)

    # delete atoms
    for incomplete_residue in incomplete_residues:
        logging.debug(
            "Deleting protein residue with incomplete sidechain "
            f"{incomplete_residue.GetName()}"
            f"{incomplete_residue.GetResidueNumber()}"
        )
        hier_view = oechem.OEHierView(processed_structure)
        structure_residue = hier_view.GetResidue(
            incomplete_residue.GetChainID(),
            incomplete_residue.GetName(),
            incomplete_residue.GetResidueNumber()
        )
        for atom in structure_residue.GetAtoms():
            processed_structure.DeleteAtom(atom)

    # spruce sometimes creates protein residues consisting of water atoms, e.g. 2hz0 chain B
    # spruce does not always delete residues with missing backbone atoms, e.g. 3qrj chain B
    # TODO: check again and submit bug report
    backbone_atom_names = {"C", "CA", "N"}
    hier_view = oechem.OEHierView(processed_structure)
    for hier_residue in hier_view.GetResidues():
        atom_names = set([atom.GetName().strip() for atom in hier_residue.GetAtoms()])
        if len(backbone_atom_names.difference(atom_names)) > 0:
            logging.debug(
                "Deleting protein residue with incomplete backbone "
                f"{hier_residue.GetResidueName()}"
                f"{hier_residue.GetResidueNumber()} ..."
            )
            for atom in hier_residue.GetAtoms():
                processed_structure.DeleteAtom(atom)

    return processed_structure


def delete_short_protein_segments(structure: oechem.OEMolBase) -> oechem.OEMolBase:
    """
    Delete protein segments consisting of 3 or less residues.

    Parameters
    ----------
    structure: oechem.OEMolBase
        An OpenEye molecule holding a protein with possibly short segments.

    Returns
    -------
    structure: oechem.OEMolBase
        An OpenEye molecule holding the protein without short segments.
    """
    # do not change input structure
    processed_structure = structure.CreateCopy()

    protein = remove_non_protein(processed_structure, remove_water=True)
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
                for atom in structure.GetAtoms(residue_predicate):
                    structure.DeleteAtom(atom)

    return structure


def delete_clashing_sidechains(
        structure: oechem.OEMolBase,
        cutoff: float = 2.0
) -> oechem.OEMolBase:
    """
    Delete side chains that are clashing with other atoms of the given structure.

    Note: Structures containing non-protein residues may lead to unexpected behavior, since also
    those residues will be deleted if clashing with other residues of the system. However, this
    behavior is important to be able to also check PTMs for clashes.

    Parameters
    ----------
    structure: oechem.OEMolBase
        An OpenEye molecule holding a protein structure.
    cutoff: float
        The distance cutoff that is used for defining a heavy atom clash.
        Note: Going bigger than 2.3 A may lead to the deletion of residues involved in strong
        hydrogen bonds.

    Returns
    -------
    processed_structure: oechem.OEMolBase
        An OpenEye molecule holding the protein structure without clashing sidechains.
    """
    from scipy.spatial import distance

    # do not change input structure
    processed_structure = structure.CreateCopy()

    # get all heavy atoms and create a KD tree for querying
    heavy_atoms = oechem.OEGraphMol()
    oechem.OESubsetMol(
        heavy_atoms,
        processed_structure,
        oechem.OEIsHeavy()
    )
    heavy_atom_coordinates_dict = heavy_atoms.GetCoords()
    heavy_atom_tree = cKDTree(list(heavy_atom_coordinates_dict.values()))
    backbone_functor = oechem.OEIsBackboneAtom(includeTerminalOxygen=True)
    # iterate over residues
    hierview = oechem.OEHierView(heavy_atoms)
    for residue in hierview.GetResidues():
        # get atom indices and coordinates
        non_backbone_atoms = [
            atom for atom in residue.GetAtoms() if not backbone_functor(atom)
        ]
        if len(non_backbone_atoms) == 0:
            # e.g. in case of a residue with missing sidechain
            continue
        non_backbone_atom_indices = [
            atom.GetIdx() for atom in non_backbone_atoms
        ]
        non_backbone_coordinates = [
            heavy_atom_coordinates_dict[index] for index in non_backbone_atom_indices
        ]
        # query for atoms close to sidechain atoms
        # this will also consider the sidechain atoms themself
        # and depending on the cutoff bonded atoms, e.g. C alphas
        non_backbone_tree = cKDTree(non_backbone_coordinates)
        potential_clashes = heavy_atom_tree.query_ball_tree(non_backbone_tree, cutoff)
        # detect number of bonds to backbone and other residues below cutoff, e.g. PTMs
        # those will also show up in the KD Tree query but are no real clash
        n_additional_clashes = 0
        for atom in non_backbone_atoms:
            for neighboring_atom in atom.GetAtoms():
                if neighboring_atom.GetIdx() not in non_backbone_atom_indices:
                    if distance.euclidean(
                        heavy_atom_coordinates_dict[atom.GetIdx()],
                        heavy_atom_coordinates_dict[neighboring_atom.GetIdx()]
                    ) < cutoff:
                        n_additional_clashes += 1
        # check if more atoms are within cutoff than number of sidechain atoms
        # as well as additional expected clashes
        if len([x for x in potential_clashes if len(x) > 0]) > \
                len(non_backbone_atom_indices) + n_additional_clashes:
            residue = residue.GetOEResidue()
            residue_name = residue.GetName()
            residue_id = residue.GetResidueNumber()
            chain_id = residue.GetChainID()
            logging.debug(
                f"Deleting clashing sidechain of {residue_name}" +
                f"{residue_id} {chain_id} ..."
            )
            # deleting sidechain atoms
            for atom in processed_structure.GetAtoms(oechem.OEAtomMatchResidue([
                f"{residue_name}:{residue_id}:.*:{chain_id}:.*:.*"
            ])):
                if not backbone_functor(atom):
                    processed_structure.DeleteAtom(atom)

    return processed_structure


def get_atom_coordinates(molecule: oechem.OEMolBase) -> List[Tuple[float, float, float]]:
    """
    Retrieve the atom coordinates of an OpenEye molecule.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule for which the coordinates should be retrieved.

    Returns
    -------
    coordinates: list of tuple of float
        The coordinates of the given molecule atoms.
    """
    coordinates_dict = molecule.GetCoords()
    # get atom coordinates in order of atom indices
    coordinates = [
        coordinates_dict[key] for key in sorted(coordinates_dict.keys())
    ]
    return coordinates


def renumber_structure(
        target_structure: oechem.OEMolBase,
        residue_ids: Iterable[int]
) -> oechem.OEGraphMol:
    """
    Renumber the residues of a protein structure according to the given list of residue IDs.

    Parameters
    ----------
    target_structure: oechem.OEMolBase
        An OpenEye molecule holding the protein structure to renumber.
    residue_ids: iterable of int
        An iterable of residue IDs matching the order of the target structure.

    Returns
    -------
    renumbered_structure: oechem.OEMolBase
        An OpenEye molecule holding the cropped protein structure.

    Raises
    ------
    ValueError
        Number of given residue IDs does not match number of residues in the given structure.
    ValueError
        Given residue IDs contain wrong types, only int is allowed.
    """
    # don't touch input structure
    renumbered_structure = target_structure.CreateCopy()

    # get residues
    hierview = oechem.OEHierView(renumbered_structure)
    structure_residues = list(hierview.GetResidues())

    # check for matching number of residues
    if len(structure_residues) != len(residue_ids):
        raise ValueError(
            "Number of given residue IDs does not match number of residues in the given " +
            "structure."
        )

    # check for matching number of residues
    if not all([isinstance(residue_id, int) for residue_id in residue_ids]):
        raise ValueError("Given residue IDs contain wrong types, only int is allowed.")

    # adjust residue numbers
    for residue_id, structure_residue in zip(residue_ids, structure_residues):
        structure_residue_mod = structure_residue.GetOEResidue()
        structure_residue_mod.SetResidueNumber(residue_id)
        for residue_atom in structure_residue.GetAtoms():
            oechem.OEAtomSetResidue(residue_atom, structure_residue_mod)

    return renumbered_structure


def superpose_proteins(
        reference_protein: oechem.OEMolBase,
        fit_protein: oechem.OEMolBase,
        residues: Iterable = tuple(),
        chain_id: str = " ",
        insertion_code: str = " "
) -> oechem.OEMolBase:
    """
    Superpose a protein structure onto a reference protein. The superposition
    can be customized to consider only the specified residues.

    Parameters
    ----------
    reference_protein: oechem.OEMolBase
        An OpenEye molecule holding a protein structure which will be used as reference during superposition.
    fit_protein: oechem.OEMolBase
        An OpenEye molecule holding a protein structure which will be superposed onto the reference protein.
    residues: Iterable of str
        Residues that should be used during superposition in format "GLY123".
    chain_id: str
        Chain identifier for residues that should be used during superposition.
    insertion_code: str
        Insertion code for residues that should be used during superposition.

    Returns
    -------
    superposed_protein: oechem.OEMolBase
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
        structure: oechem.OEMolBase,
        keep_protein_residue_ids: bool = True,
        keep_chain_ids: bool = False,
) -> oechem.OEMolBase:
    """
    Update the atom, residue and chain IDs of the given molecular structure. All residues become
    part of chain A, unless 'keep_chain_ids' is set True. Atom IDs will start from 1. Residue IDs
    will start from 1, except 'keep_protein_residue_ids' is set True. This is especially useful, if
    molecules were merged, which can result in overlapping atom and residue IDs as well as
    separate chains.

    Parameters
    ----------
    structure: oechem.OEMolBase
        The OpenEye molecule structure for updating atom and residue ids.
    keep_protein_residue_ids: bool
        If the protein residues should be kept.
    keep_chain_ids: bool
        If the chain IDS should be kept.

    Returns
    -------
    structure: oechem.OEMolBase
        The OpenEye molecule structure with updated atom and residue ids.
    """
    # update residue identifiers, except for residue number,
    # residue names, atom names, chain id, record type, insert code,
    # alternate location
    preserved_info = (
            oechem.OEPreserveResInfo_ResidueNumber
            | oechem.OEPreserveResInfo_ResidueName
            | oechem.OEPreserveResInfo_AtomName
            | oechem.OEPreserveResInfo_ChainID
            | oechem.OEPreserveResInfo_HetAtom
            | oechem.OEPreserveResInfo_InsertCode
            | oechem.OEPreserveResInfo_AlternateLocation
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
                if not keep_chain_ids:
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
                if not keep_chain_ids:
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
                if not keep_chain_ids:
                    residue.SetChainID("A")
                residue.SetResidueNumber(residue_number)
                oechem.OEAtomSetResidue(atom, residue)

    # order atoms into PDB order
    oechem.OEPDBOrderAtoms(structure)

    # update residue identifiers, except for residue number,
    # residue names, atom names, chain id, record type, insert code,
    # alternate location
    preserved_info = (
            oechem.OEPreserveResInfo_ResidueNumber
            | oechem.OEPreserveResInfo_ResidueName
            | oechem.OEPreserveResInfo_AtomName
            | oechem.OEPreserveResInfo_ChainID
            | oechem.OEPreserveResInfo_HetAtom
            | oechem.OEPreserveResInfo_InsertCode
            | oechem.OEPreserveResInfo_AlternateLocation
    )
    oechem.OEPerceiveResidues(structure, preserved_info)

    return structure


def split_molecule_components(molecule: oechem.OEMolBase) -> List[oechem.OEGraphMol]:
    """
    Split an OpenEye molecule into its bonded components.

    Parameters
    ----------
    molecule: oechem.OEMolBase
        An OpenEye molecule holding multiple components.

    Returns
    -------
    components: list of oechem.OEGraphMol
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


def residue_ids_to_residue_names(
        structure: oechem.OEMolBase,
        residue_ids: List[int],
        chain_id: Union[None, str] = None
) -> List[str]:
    """
    Get the corresponding residue names for a list of residue IDs and a give OpenEye molecule
    with residue information.

    Parameters
    ----------
    structure: oechem.OEMolBase
        An OpenEye molecule with residue information.
    residue_ids: list of int
        A list of residue IDs.
    chain_id: None or str
        The chain ID to filter for.

    Returns
    -------
    residue_names: list of str
        The corresponding residue names as three letter codes.

    Raises
    ------
    ValueError
        No residue found for residue ID {resid}.
    ValueError
        Found multiple residues for residue ID {resid}.
    """
    residue_names = []
    if not chain_id:
        chain_id = ".*"
    for resid in residue_ids:
        predicate = oechem.OEAtomMatchResidue(f".*:{resid}:.*:{chain_id}:.*:.*")
        selection_residue_names = set([
            oechem.OEAtomGetResidue(atom).GetName() for atom in structure.GetAtoms(predicate)
        ])
        if len(selection_residue_names) == 0:
            raise ValueError(f"No residue found for residue ID {resid}.")
        elif len(selection_residue_names) > 1:
            raise ValueError(f"Found multiple residues for residue ID {resid}.")
        residue_names.append(selection_residue_names.pop())

    return residue_names
