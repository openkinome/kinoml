from typing import List, Union

from openeye import oechem, oegrid, oespruce


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


def _set_design_unit_options(
    loop_db: Union[str, None] = None,
    cap_C_termini: bool = True,
    cap_N_termini: bool = True,
) -> oespruce.OEMakeDesignUnitOptions:
    """
    Set options for design unit generation.
    Parameters
    ----------
    loop_db: str or None
        Path to OpenEye Spruce loop database.
    cap_C_termini: bool
        If C termini should be capped with ACE.
    cap_N_termini: bool
        If N termini should be capped with NME.
    Returns
    -------
    design_unit_options: oespruce.OEMakeDesignUnitOptions
        An OpenEye class holding options for design unit generation.
    """
    from pathlib import Path

    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    if loop_db is not None:
        loop_db = str(Path(loop_db).expanduser().resolve())
        design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    if cap_C_termini:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(True)
    else:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)

    if cap_N_termini:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(True)
    else:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)

    return design_unit_options


def prepare_complex(
    protein_ligand_complex: oechem.OEGraphMol,
    electron_density: Union[oegrid.OESkewGrid, None] = None,
    loop_db: Union[str, None] = None,
    cap_C_termini: bool = True,
    cap_N_termini: bool = True,
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
    cap_C_termini: bool
        If C termini should be capped with ACE.
    cap_N_termini: bool
        If N termini should be capped with NME.
    Returns
    -------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a prepared protein structure.
    ligand: oechem.OEGraphMol
        An OpenEye molecule holding a prepared ligand structure.
    """

    # set design unit options
    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = _set_design_unit_options(
        loop_db, cap_C_termini, cap_N_termini
    )

    # make design units
    if electron_density is None:
        design_units = list(
            oespruce.OEMakeDesignUnits(
                protein_ligand_complex, structure_metadata, design_unit_options
            )
        )
    else:
        design_units = list(
            oespruce.OEMakeDesignUnits(
                protein_ligand_complex,
                electron_density,
                structure_metadata,
                design_unit_options,
            )
        )
    if len(design_units) == 1:
        design_unit = design_units[0]
    elif len(design_units) > 1:
        design_unit = design_units[0]
    else:
        # TODO: Returns list of Nones if something goes wrong
        return [None, None]

    # get protein
    protein = oechem.OEGraphMol()
    design_unit.GetProtein(protein)

    # add missing OXT backbone atoms, not handled by OEFixBackbone in OESpruce 1.1.0
    for atom in protein.GetAtoms():
        if "H'" in atom.GetName():
            atom.SetAtomicNum(8)
            atom.SetName("OXT")
            atom.SetFormalCharge(-1)

    # get ligand
    ligand = oechem.OEGraphMol()
    design_unit.GetLigand(ligand)

    return [protein, ligand]


def prepare_protein(
    protein: oechem.OEGraphMol,
    loop_db: Union[str, None] = None,
    cap_C_termini: bool = True,
    cap_N_termini: bool = True,
) -> Union[oechem.OEGraphMol, None]:
    """
    Prepare an OpenEye molecule holding a protein structure for docking.
    Parameters
    ----------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a structure with protein.
    loop_db: str
        Path to OpenEye Spruce loop database.
    cap_C_termini: bool
        If C termini should be capped with ACE.
    cap_N_termini: bool
        If N termini should be capped with NME.
    Returns
    -------
    protein: oechem.OEGraphMol
        An OpenEye molecule holding a prepared protein structure.
    """

    # set bio design unit options
    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = _set_design_unit_options(
        loop_db, cap_C_termini, cap_N_termini
    )

    # make bio design units
    bio_design_units = list(
        oespruce.OEMakeBioDesignUnits(protein, structure_metadata, design_unit_options)
    )
    if len(bio_design_units) == 1:
        bio_design_unit = bio_design_units[0]
    elif len(bio_design_units) > 1:
        bio_design_unit = bio_design_units[0]
    else:
        # TODO: Returns None if something goes wrong
        return None

    # get protein
    protein = oechem.OEGraphMol()
    bio_design_unit.GetProtein(protein)

    # add missing OXT backbone atoms, not handled by OEFixBackbone in OESpruce 1.1.0
    for atom in protein.GetAtoms():
        if "H'" in atom.GetName():
            atom.SetAtomicNum(8)
            atom.SetName("OXT")
            atom.SetFormalCharge(-1)

    return protein
