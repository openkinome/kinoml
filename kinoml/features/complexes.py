"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
from __future__ import annotations
from functools import lru_cache
from typing import List, Union

from appdirs import user_cache_dir

from .core import BaseFeaturizer
from ..core.ligands import FileLigand
from ..core.proteins import FileProtein
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
        ligands = self._read_ligands(system.ligand.path)
        protein = self._read_protein(system.protein.path)

        # TODO: more sophisticated decision between hybrid and chemgauss docking
        if self._has_ligand(protein):

            # TODO: electron density, loop database
            prepared_protein, prepared_ligand = self._prepare_complex(protein)
            hybrid_receptor = self._create_hybrid_receptor(
                prepared_protein, prepared_ligand
            )
            docking_poses = self._hybrid_docking(hybrid_receptor, ligands)

        else:
            # TODO: identify pocket with point or box
            # TODO: check possibility to define design unit with residue (would consider electron density)
            raise NotImplemented

        # TODO: where to store data
        protein_path = f"{user_cache_dir()}/{system.protein.name}.pdb"  # mmcif writing not supported by openeye
        self._write_molecules([prepared_protein], protein_path)
        file_protein = FileProtein(path=protein_path)

        ligand_path = f"{user_cache_dir()}/{system.ligand.name}.sdf"
        self._write_molecules(docking_poses, ligand_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )

        return protein_ligand_complex

    @staticmethod
    def _read_ligands(path: str) -> List[oechem.OEGraphMol]:
        """
        Read ligands from a file.
        Parameters
        ----------
        path: str
            Path to ligand file.
        Returns
        -------
        molecules: list of oechem.OEGraphMol
            A List of ligands as OpenEye molecules.
        """
        from openeye import oechem

        molecules = []
        with oechem.oemolistream(path) as ifs:
            for molecule in ifs.GetOEGraphMols():
                molecules.append(oechem.OEGraphMol(molecule))

        # TODO: returns empty list if something goes wrong
        return molecules

    @staticmethod
    def _read_protein(path: str) -> oechem.OEGraphMol:
        """
        Read a protein from a file.
        Parameters
        ----------
        path: str
            Path to protein file.
        Returns
        -------
        molecule: oechem.OEGraphMol
            A protein as OpenEye molecule.
        """
        from openeye import oechem

        suffix = path.split(".")[-1]
        with oechem.oemolistream(path) as ifs:
            if suffix == "pdb":
                ifs.SetFlavor(
                    oechem.OEFormat_PDB,
                    oechem.OEIFlavor_PDB_Default
                    | oechem.OEIFlavor_PDB_DATA
                    | oechem.OEIFlavor_PDB_ALTLOC,
                )
            elif suffix == "cif":
                ifs.SetFlavor(oechem.OEFormat_MMCIF, oechem.OEIFlavor_MMCIF_Default)
            elif suffix == "mol":
                ifs.SetFlavor(oechem.OEFormat_MDL, oechem.OEIFlavor_MDL_Default)
            elif suffix == "mol2":
                ifs.SetFlavor(oechem.OEFormat_MOL2, oechem.OEIFlavor_MOL2_Default)
            elif suffix == "xyz":
                ifs.SetFlavor(oechem.OEFormat_XYZ, oechem.OEIFlavor_XYZ_Default)
            # TODO: add reasonable defaults for other file formats

            molecule = oechem.OEGraphMol()
            oechem.OEReadMolecule(ifs, molecule)

        # TODO: returns molecule with 0 atoms if something goes wrong
        return molecule

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
            print("More than one design unit found---using first one")
            bio_design_unit = bio_design_units[0]
        else:
            print("No design units found")
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
    def _create_box_receptor(
        protein: oechem.OEGraphMol,
        xmax: Union[float, int],
        ymax: Union[float, int],
        zmax: Union[float, int],
        xmin: Union[float, int],
        ymin: Union[float, int],
        zmin: Union[float, int],
    ) -> oechem.OEGraphMol:
        """
        Create a box receptor for docking.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding a prepared protein structure.
        xmax: float or int
            Maximal number in x direction.
        ymax: float or int
            Maximal number in y direction.
        zmax: float or int
            Maximal number in z direction.
        xmin: float or int
            Minimal number in x direction.
        ymin: float or int
            Minimal number in x direction.
        zmin: float or int
            Minimal number in z direction.
        Returns
        -------
        receptor: oechem.OEGraphMol
            An OpenEye molecule holding a receptor with defined binding site via box dimensions.
        """
        from openeye import oechem, oedocking

        # create receptor
        box = oedocking.OEBox(xmax, ymax, zmax, xmin, ymin, zmin)
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
