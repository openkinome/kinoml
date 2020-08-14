"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
from __future__ import annotations
from functools import lru_cache
from typing import List, Union, Tuple

from appdirs import user_cache_dir

from .core import BaseFeaturizer
from ..core.ligands import FileLigand
from ..core.proteins import FileProtein, PDBProtein
from ..core.systems import ProteinLigandComplex


class OpenEyesProteinLigandDockingFeaturizer(BaseFeaturizer):

    """
    Given a System with exactly one protein and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that a file-based System object will be passed; this
    means we will have a System.components with (FileProtein, FileLigand).
    The file itself could be a URL.
    """

    from openeye import oechem, oegrid  # only needed for typing

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    @lru_cache(maxsize=100)
    def _featurize(self, system: ProteinLigandComplex) -> ProteinLigandComplex:
        """
        Perform docking with OpenEye toolkits and thoughtful defaults.
        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding protein and ligand information.
        Returns
        -------
        protein_ligand_complex: ProteinLigandComplex
            The same system but with docked ligand.
        """
        ligands = self._read_molecules(system.ligand.path)
        protein = self._read_molecules(system.protein.path)[0]
        # TODO: electron density might be redundant here, if already used in separate protein preparation workflow
        if system.protein.electron_density_path is not None:
            electron_density = self._read_electron_density(
                system.protein.electron_density_path
            )
        else:
            electron_density = None

        # TODO: more sophisticated decision between hybrid and chemgauss docking
        if self._has_ligand(protein):

            # TODO: electron density, loop database
            prepared_protein, prepared_ligand = self._prepare_complex(
                protein, electron_density
            )
            hybrid_receptor = self._create_hybrid_receptor(
                prepared_protein, prepared_ligand
            )
            docking_poses = self._hybrid_docking(hybrid_receptor, ligands)
        else:
            if isinstance(system.protein, PDBProtein):
                # TODO: check possibility to define design unit with residue (would consider electron density)
                prepared_protein = self._prepare_protein(protein)
                klifs_pocket = PDBProtein.klifs_pocket(
                    system.protein.pdb_id
                )  # TODO: specify chain and altloc
                box_dimensions = self._resids_to_box(protein, klifs_pocket)
                box_receptor = self._create_box_receptor(protein, box_dimensions)
                docking_poses = self._chemgauss_docking(box_receptor, ligands)
            else:
                raise NotImplemented

        # TODO: where to store data
        protein_path = f"{user_cache_dir()}/{system.protein.name}.pdb"  # mmcif writing not supported by openeye
        self._write_molecules([prepared_protein], protein_path)
        file_protein = FileProtein(path=protein_path)

        ligand_path = (
            f"{user_cache_dir()}/{system.protein.name}_{system.ligand.name}.sdf"
        )
        self._write_molecules(docking_poses, ligand_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )

        return protein_ligand_complex

    @staticmethod
    def _read_molecules(path: str) -> List[oechem.OEGraphMol]:
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
        from openeye import oechem

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

    @staticmethod
    def _read_electron_density(path: str) -> Union[oegrid.OESkewGrid, None]:
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
        from openeye import oegrid

        electron_density = oegrid.OESkewGrid()
        # TODO: different map formats
        if not oegrid.OEReadMTZ(path, electron_density, oegrid.OEMTZMapType_Fwt):
            electron_density = None
        return electron_density

    @staticmethod
    def _write_molecules(molecules: List[oechem.OEGraphMol], path: str):
        """
        Save molecules to file.
        Parameters
        ----------
        molecules: list of oechem.OEGraphMol
            A list of OpenEye molecules for writing.
        path: str
            File path for saving molecules.
        """
        from openeye import oechem

        with oechem.oemolostream(path) as ofs:
            for molecule in molecules:
                oechem.OEWriteMolecule(ofs, molecule)
        return

    @staticmethod
    def _has_ligand(molecule: oechem.OEGraphMol) -> bool:
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
        from openeye import oechem

        ligand = oechem.OEGraphMol()
        protein = oechem.OEGraphMol()
        water = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        oechem.OESplitMolComplex(ligand, protein, water, other, molecule)

        if ligand.NumAtoms() > 0:
            return True

        return False

    @staticmethod
    def _prepare_complex(
        protein_ligand_complex: oechem.OEGraphMol,
        electron_density: Union[oegrid.OESkewGrid, None] = None,
        loop_db: Union[str, None] = None,
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
        Returns
        -------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a prepared protein structure.
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding a prepared ligand structure.
        """
        from openeye import oechem, oespruce

        # create design units
        structure_metadata = oespruce.OEStructureMetadata()
        design_unit_options = oespruce.OEMakeDesignUnitOptions()
        if loop_db is not None:
            design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
                loop_db
            )
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

        # get ligand
        ligand = oechem.OEGraphMol()
        design_unit.GetLigand(ligand)

        return [protein, ligand]

    @staticmethod
    def _prepare_protein(
        protein: oechem.OEGraphMol, loop_db: Union[str, None] = None
    ) -> Union[oechem.OEGraphMol, None]:
        """
        Prepare an OpenEye molecule holding a protein structure for docking.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a structure with protein.
        loop_db: str
            Path to OpenEye Spruce loop database.
        Returns
        -------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a prepared protein structure.
        """
        from openeye import oechem, oespruce

        # create bio design units
        structure_metadata = oespruce.OEStructureMetadata()
        design_unit_options = oespruce.OEMakeDesignUnitOptions()
        if loop_db is not None:
            design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
                loop_db
            )
        bio_design_units = list(
            oespruce.OEMakeBioDesignUnits(
                protein, structure_metadata, design_unit_options
            )
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

        return protein

    @staticmethod
    def _create_hybrid_receptor(
        protein: oechem.OEGraphMol, ligand: oechem.OEGraphMol
    ) -> oechem.OEGraphMol:
        """
        Create a receptor for hybrid docking.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a prepared protein structure.
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding a prepared ligand structure.
        Returns
        -------
        receptor: oechem.OEGraphMol
            An OpenEye molecule holding a receptor with protein and ligand.
        """
        from openeye import oechem, oedocking

        # create receptor
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, ligand)

        return receptor

    @staticmethod
    def _create_hint_receptor(
        protein: oechem.OEGraphMol,
        hintx: Union[float, int],
        hinty: Union[float, int],
        hintz: Union[float, int],
    ) -> oechem.OEGraphMol:
        """
        Create a hint receptor for docking.
        Parameters
        ----------
        protein: oechem.OEGraphMol
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
        from openeye import oechem, oedocking

        # create receptor
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, hintx, hinty, hintz)

        return receptor

    @staticmethod
    def _resids_to_box(
        protein: oechem.OEGraphMol, resids: List[int]
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Retrieve box dimensions of a list if residues.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a protein structure.
        resids: list of int
            A list of resids defining the residues of interest.
        Returns
        -------
        box_dimensions: tuple of float
            The box dimensions in the order of xmax, ymax, zmax, xmin, ymin, zmin.
        """
        from openeye import oechem

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

    @staticmethod
    def _create_box_receptor(
        protein: oechem.OEGraphMol,
        box_dimensions: Tuple[float, float, float, float, float, float],
    ) -> oechem.OEGraphMol:
        """
        Create a box receptor for docking.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a prepared protein structure.
        box_dimensions: tuple of float
            The box dimensions in the order of xmax, ymax, zmax, xmin, ymin, zmin.
        Returns
        -------
        receptor: oechem.OEGraphMol
            An OpenEye molecule holding a receptor with defined binding site via box dimensions.
        """
        from openeye import oechem, oedocking

        # create receptor
        box = oedocking.OEBox(*box_dimensions)
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, box)

        return receptor

    @staticmethod
    def _run_docking(
        receptor: oechem.OEGraphMol,
        molecules: List[oechem.OEGraphMol],
        dock_method: int,
        num_poses: int = 1,
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
        Returns
        -------
        docked_molecules: list of oechem.OEGraphMol
            A list of OpenEye molecules holding the docked molecules.
        """
        from openeye import oechem, oedocking, oequacpac, oeomega

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
            # enumerate tautomers
            tautomer_options = oequacpac.OETautomerOptions()
            tautomer_options.SetMaxTautomersGenerated(4096)
            tautomer_options.SetMaxTautomersToReturn(16)
            tautomer_options.SetCarbonHybridization(True)
            tautomer_options.SetMaxZoneSize(50)
            tautomer_options.SetApplyWarts(True)
            pKa_norm = True
            tautomers = [
                oechem.OEMol(tautomer)
                for tautomer in oequacpac.OEGetReasonableTautomers(
                    molecule, tautomer_options, pKa_norm
                )
            ]

            # set up omega
            omega_options = oeomega.OEOmegaOptions()
            omega_options.SetMaxSearchTime(60.0)  # time out
            omega = oeomega.OEOmega(omega_options)
            omega.SetStrictStereo(False)  # enumerate stereochemistry if uncertain

            docked_tautomers = list()
            # dock tautomers
            for mol in tautomers:
                docked_mol = oechem.OEMol()
                # expand conformers
                omega.Build(mol)

                # dock molecule
                return_code = dock.DockMultiConformerMolecule(
                    docked_mol, mol, num_poses
                )
                if return_code != oedocking.OEDockingReturnCode_Success:
                    # TODO: Maybe something for metadata
                    print(
                        f"Docking failed for molecule with title {mol.GetTitle()} with error code "
                        f"{oedocking.OEDockingReturnCodeGetName(return_code)}."
                    )
                    continue

                # store docking data
                oedocking.OESetSDScore(docked_mol, dock, dock.GetName())

                # expand conformations
                for conformation in docked_mol.GetConfs():
                    docked_tautomers.append(oechem.OEGraphMol(conformation))

            # sort all conformations of all tautomers by score
            docked_tautomers.sort(key=score)

            # keep number of conformations as specified by num_poses
            docked_molecules += docked_tautomers[:num_poses]

        if len(docked_molecules) == 0:
            # TODO: returning None when something goes wrong
            return None

        return docked_molecules

    @classmethod
    def _hybrid_docking(
        cls,
        hybrid_receptor: oechem.OEGraphMol,
        molecules: List[oechem.OEGraphMol],
        num_poses: int = 1,
    ) -> Union[List[oechem.OEGraphMol], None]:
        """
        Dock molecules into a prepared receptor holding protein and ligand structure.
        Parameters
        ----------
        hybrid_receptor: oechem.OEGraphMol
            An OpenEye molecule holding the prepared receptor.
        molecules: list of oechem.OEGraphMol
            A list of OpenEye molecules holding prepared molecules for docking.
        num_poses: int
            Number of docking poses to generate per molecule.
        Returns
        -------
        docked_molecules: list of oechem.OEGraphMol
            A list of OpenEye molecules holding the docked molecules.
        """
        from openeye import oedocking

        dock_method = oedocking.OEDockMethod_Hybrid2
        docked_molecules = cls._run_docking(
            hybrid_receptor, molecules, dock_method, num_poses
        )

        return docked_molecules

    @classmethod
    def _chemgauss_docking(
        cls,
        receptor: oechem.OEGraphMol,
        molecules: List[oechem.OEGraphMol],
        num_poses: int = 1,
    ) -> Union[List[oechem.OEGraphMol], None]:
        """
        Dock molecules into a prepared receptor holding a protein structure.
        Parameters
        ----------
        receptor: oechem.OEGraphMol
            An OpenEye molecule holding the prepared hint or box receptor.
        molecules: list of oechem.OEGraphMol
            A list of OpenEye molecules holding prepared molecules for docking.
        num_poses: int
            Number of docking poses to generate per molecule.
        Returns
        -------
        docked_molecules: list of oechem.OEGraphMol
            A list of OpenEye molecules holding the docked molecules.
        """
        from openeye import oedocking

        dock_method = oedocking.OEDockMethod_Chemgauss4
        docked_molecules = cls._run_docking(receptor, molecules, dock_method, num_poses)

        return docked_molecules
