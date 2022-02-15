import logging
from pathlib import Path
from typing import Iterable, Tuple, Union

from MDAnalysis.core.universe import Merge, Universe
from MDAnalysis.core.groups import AtomGroup, Residue


logger = logging.getLogger(__name__)


def read_molecule(path: Union[str, Path]) -> Universe:
    """
    Read a molecule from a file. Uses Biopython to support reading of the CIF format.

    Parameters
    ----------
    path: str, pathlib.Path
        Path to molecule file.

    Returns
    -------
    molecule: MDAnalysis.core.universe.Universe
        The MDAnalysis universe.
    """
    import MDAnalysis as mda

    path = str(Path(path).expanduser().resolve())
    suffix = path.split(".")[-1]
    if suffix == "cif":
        from tempfile import NamedTemporaryFile
        from Bio.PDB import MMCIFParser, PDBIO

        parser = MMCIFParser()
        structure = parser.get_structure("", path)
        with NamedTemporaryFile(suffix="pdb") as tempfile:
            io = PDBIO()
            io.set_structure(structure)
            io.save(tempfile.name)
            molecule = mda.Universe(tempfile.name, in_memory=True, dt=0, guess_bonds=True)
    else:
        molecule = mda.Universe(path, in_memory=True, dt=0, guess_bonds=True)

    return molecule


def select_chain(molecule: Union[Universe, AtomGroup], chain_id: str) -> Universe:
    """
    Select a chain from an MDAnalysis molecule.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a molecular structure.
    chain_id: str
        Chain identifier.

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        An MDAnalysis universe holding the selected chain.
    """
    selection = molecule.select_atoms(f"chainID {chain_id}")
    return Merge(selection)


def select_altloc(
        molecule: Union[Universe, AtomGroup],
        altloc_id: str,
        altloc_fallback: bool = True,
) -> Universe:
    """
    Select an alternate location from an MDAnalysis molecule.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a molecular structure.
    altloc_id: str
        Alternate location identifier.
    altloc_fallback: bool
        If the alternate location "A" should be used for residues that do
        not contain the given alternate location identifier.

    Returns
    -------
    selection: MDAnalysis.core.universe.Universe
        An MDAnalysis universe holding the selected alternate location.

    Raises
    ------
    ValueError
        No atoms were found with given altloc id.
    """
    import itertools

    # find all atoms with alternate locations
    altloc_register = {}
    for atom in molecule.atoms:
        if atom.altLoc:
            atom_details = (atom.chainID, atom.resname, atom.resid, atom.name)
            if atom_details not in altloc_register.keys():
                altloc_register[atom_details] = []
            altloc_register[atom_details].append(atom.altLoc)

    # check if alternate location of interest is actually present
    if altloc_id not in set(itertools.chain.from_iterable(altloc_register.values())):
        raise ValueError("No atoms were found with given altloc id.")

    # define atoms with alternate locations to exclude
    altloc_exclusion = []
    for atom_details, found_altlocs in altloc_register.items():
        if altloc_id in found_altlocs:
            for found_altloc in found_altlocs:
                if found_altloc != altloc_id:
                    altloc_exclusion.append(atom_details + (found_altloc,))
        else:
            if altloc_fallback:  # go for altloc "A"
                for found_altloc in found_altlocs:
                    if found_altloc != "A":
                        altloc_exclusion.append(atom_details + (found_altloc,))

    selection_command = "not (" + " or ".join([
        f"(chainID {chain_id} and resname {resname} and resid {resid} and name {name} "
        f"and altLoc {altloc})" for chain_id, resname, resid, name, altloc in altloc_exclusion
    ]) + ")"

    selection = molecule.select_atoms(selection_command)
    return Merge(selection)


def remove_non_protein(
    molecule: Union[Universe, AtomGroup],
    exceptions: Union[None, Iterable[str]] = None,
    only_standard_amino_acids: bool = True,
    remove_water: bool = False,
) -> Universe:
    """
    Remove non-protein atoms from an OpenEye molecule. Water will be kept by default.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a molecular structure.
    exceptions: None or iterable of str
        Exceptions that should not be removed.
    only_standard_amino_acids: bool, default=True
        If only standard amino acids shell be retained, .i.e. ALA, ARG, ASN, ASP, CYS, GLN, GLU,
        GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SEC, SER, THR, TRP, TYR, VAL.
    remove_water: bool, default=False
        If water should be removed.

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        An MDAnalysis universe holding the filtered structure.
    """
    # add protein to selection command
    standard_amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SEC", "SER", "THR", "TRP", "TYR", "VAL"
    ]
    if only_standard_amino_acids:
        selection_command = " or ".join([
                f"resname {resname}" for resname in standard_amino_acids
            ])
    else:
        selection_command = "protein or resname NMA"
    # add water and exceptions to selection command
    if exceptions is None:
        exceptions = []
    if remove_water is False:
        exceptions.append("HOH")
    if len(exceptions) > 0:
        selection_command = selection_command + " or " + " or ".join([
            f"resname {resname}" for resname in exceptions
        ])

    selection = molecule.select_atoms(selection_command)
    return Merge(selection)


def delete_residues(molecule: Universe, residues: Iterable[Residue]):
    """
    Delete residues from an MDAnalysis molecule.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a molecular structure.
    residues: iterable of MDAnalysis.core.groups.Residue

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        An MDAnalysis molecule holding a molecular structure without given residues.
    """
    selection_command = "not (" + " or ".join([
        f"(resname {residue.resname} and resid {residue.resid} and chainID {residue.segid})"
        for residue in residues
    ]) + ")"
    selection = molecule.select_atoms(selection_command)

    return Merge(selection)


def delete_expression_tags(
        molecule: Union[Universe, AtomGroup], pdb_path: Union[str, Path]
) -> Universe:
    """
    Delete expression tags listed in the PDB header section "SEQADV".

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule.
    pdb_path: str or pathlib.Path
        The path to the PDB file containing the information about expression tags.

    Returns
    -------
    : Universe
        The MDAnalysis universe without expression tags.
    """
    expression_tags = []
    with open(pdb_path, "r") as pdb_file:
        for line in pdb_file.readlines():
            if line.startswith("SEQADV"):
                if "EXPRESSION TAG" in line or "CLONING ARTIFACT" in line:
                    expression_tags.append(line.split()[2:5])

    if len(expression_tags) == 0:
        return molecule

    selection_command = "not (" + " or ".join([
        f"(resname {resname} and resid {resid} and chainID {chain_id})"
        for resname, chain_id, resid in expression_tags
    ]) + ")"
    selection = molecule.select_atoms(selection_command)

    return Merge(selection)


def get_sequence(molecule: Union[Universe, AtomGroup]) -> str:
    """
    Get the amino acid sequence with one letter characters of an MDAnalysis molecule.
    All residues not named as standard amino acid will receive the character 'X'.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule.

    Returns
    -------
    sequence: str
        The amino acid sequence with one letter characters.
    """
    aa_dict = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
        "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
        "PRO": "P", "SEC": "U", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }

    sequence = []

    for resname in molecule.residues.resnames:
        try:
            sequence.append(aa_dict[resname])
        except KeyError:
            sequence.append("X")

    sequence = "".join(sequence)

    return sequence


def get_structure_sequence_alignment(
    molecule: Union[Universe, AtomGroup], sequence: str
) -> Tuple[str, str]:
    """
    Generate an alignment between an MDAnalysis protein structure and an amino acid sequence. The
    provided protein structure should only contain protein residues to prevent unexpected behavior.
    Also, this alignment was optimized for highly similar sequences, i.e. only few mutations,
    deletions and insertions. Non protein residues will be marked with "X". The provided MDAnalysis
    molecule needs bond information to properly work, hence use "guess_bonds=True" when loading
    the universe.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a protein structure.
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
        """Check if two MDAnalysis residues are connected."""
        residue1_atom_indices = residue1.atoms.indices
        for atom in residue2.atoms:
            for bonded_atom in atom.bonded_atoms:
                if bonded_atom.index in residue1_atom_indices:
                    return True
        return False

    # align template and target sequences
    target_sequence = get_sequence(molecule)
    sequence_aligned, structure_sequence_aligned = pairwise2.align.globalxs(
        sequence, target_sequence, open=-1, extend=0
    )[0][:2]

    # correct alignments involving gaps
    structure_residues = list(molecule.residues)
    gaps = re.finditer("[^-][-]+[^-]", structure_sequence_aligned)
    for gap in gaps:
        gap_start = gap.start() - structure_sequence_aligned[: gap.start() + 1].count("-")
        start_residue = structure_residues[gap_start - 1]
        end_residue = structure_residues[gap_start]
        gap_sequence = sequence_aligned[gap.start(): gap.end() - 2]
        # check for connected residues, which could indicate a wrong alignment
        # e.g. ABEDEFG     ABEDEFG
        #      ABE--FG <-> AB--EFG
        if _connected_residues(structure_residues[gap_start], structure_residues[gap_start + 1]):
            # check if gap involves last residue but is connected
            if gap.end() == len(structure_sequence_aligned):
                structure_sequence_aligned = (
                    structure_sequence_aligned[: gap.start() + 1]
                    + gap.group()[1:][::-1]
                    + structure_sequence_aligned[gap.end():]
                )
            else:
                # check two ways to invert gap
                if not _connected_residues(
                    structure_residues[gap_start - 1], structure_residues[gap_start]
                ):
                    # i.e. ABEDEFG     ABEDEFG
                    #      ABE--FG --> AB--EFG
                    structure_sequence_aligned = (
                        structure_sequence_aligned[: gap.start()]
                        + gap.group()[:-1][::-1]
                        + structure_sequence_aligned[gap.end() - 1 :]
                    )
                elif not _connected_residues(
                    structure_residues[gap_start + 1], structure_residues[gap_start + 2]
                ):
                    # i.e. ABEDEFG     ABEDEFG
                    #      AB--EFG --> AB--EFG
                    structure_sequence_aligned = (
                        structure_sequence_aligned[: gap.start() + 1]
                        + gap.group()[1:][::-1]
                        + structure_sequence_aligned[gap.end():]
                    )
                else:
                    # i.e. ABEDEFG     ABEDEFG
                    #      AB**EFG --> AB--EFG
                    logging.debug(
                        f"Alignment contains insertion with sequence {gap_sequence}"
                        + f" between bonded residues {start_residue.resid}"
                        + f" and {end_residue.resid}, "
                        + "keeping original alignment ..."
                    )
                    continue
            logger.debug("Corrected sequence gap ...")

    return structure_sequence_aligned, sequence_aligned


def delete_alterations(
        molecule: Union[Universe, AtomGroup],
        sequence: str,
        delete_n_anchors: int = 2,
        short_protein_segments_cutoff: int = 3,
) -> Universe:
    """
    Delete residues from an MDAnalysis molecule that are not covered by the given sequence, i.e.
    mutations and insertions. The provided protein structure should only contain protein residues
    to prevent unexpected behavior.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a protein structure.
    sequence: str
        A one letter amino acid sequence.
    delete_n_anchors: int, default=2
        The number of anchoring residues to delete when deleting insertions.
    short_protein_segments_cutoff: int, default=3
        If deleting residues will lead to short protein segments, delete these segments with a
        length up to the specified cutoff.

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        An MDAnalysis molecule holding a protein structure without alterations deleted.
    """
    import re

    target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
        molecule, sequence
    )
    residues = list(molecule.residues)
    residues_to_delete = set()
    # delete substitutions and insertions
    target_residue_counter = 0
    for target_sequence_residue, template_sequence_residue in zip(
            target_sequence_aligned, template_sequence_aligned
    ):
        if target_sequence_residue != "-":
            target_residue_counter += 1
            if template_sequence_residue != target_sequence_residue:
                residues_to_delete.add(residues[target_residue_counter - 1])

    # delete anchoring residues of insertions if required
    if delete_n_anchors > 0:
        terminus_length_cutoff = delete_n_anchors + short_protein_segments_cutoff
        regex = (
            # insertion at beginning considering anchoring sequence and short protein segments
            "^[^-]{1," + str(terminus_length_cutoff) + "}[-]+"
            # insertions within sequence considering anchoring sequence and short protein segments
            "|(?<=[^-]{" + str(short_protein_segments_cutoff + 1) + "})"  # not too close at begin
            "[^-]{" + str(delete_n_anchors) + "}[-]+[^-]{" + str(delete_n_anchors) + "}"  # match
            + "(?=[^-]{" + str(short_protein_segments_cutoff + 1) + "})"  # not too close at end
            # insertion at the end considering anchoring sequence and short protein segments
            "|[-]+[^-]{1," + str(terminus_length_cutoff) + "}$"
        )
        insertions = re.finditer(regex, template_sequence_aligned)
        for insertion in insertions:
            insertion_start = insertion.start() - \
                              target_sequence_aligned[: insertion.start()].count("-")
            insertion_end = insertion.end() - target_sequence_aligned[: insertion.end()].count("-")
            residues_to_delete.update(molecule.residues[insertion_start:insertion_end])

    if len(residues_to_delete) > 0:
        residues_to_delete_text = ", ".join([
            f"{residue.resname}{residue.resid}{residue.segid}" for residue in residues_to_delete
        ])
        logger.debug(
            f"Deleting alterations: {residues_to_delete_text}"
        )
        molecule = delete_residues(molecule, residues_to_delete)

    if short_protein_segments_cutoff > 0:
        molecule = delete_short_protein_segments(molecule, short_protein_segments_cutoff)

    return Merge(molecule.atoms)


def delete_short_protein_segments(
        molecule: Union[Universe, AtomGroup], cutoff: int = 3
) -> Universe:
    """
    Delete protein segments consisting of 3 or less residues. Needs to have bonding information
    to work correctly.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a protein with possibly short segments.
    cutoff: int, default=3
        The upper bound defining a short protein segment.

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        An MDAnalysis molecule holding the protein without short segments.
    """
    protein = molecule.select_atoms("protein")

    segments = []
    segment = []
    for residue in protein.residues:
        n_peptide_bonds = sum([
            1 for bond in residue.get_connections(typename="bonds", outside=True)
            if tuple(bond.atoms.names) == ("C", "N")  # peptide bond
        ])
        if n_peptide_bonds == 0:
            segments.append([residue])
        else:
            segment.append(residue)
        if n_peptide_bonds == 1 and len(segment) > 1:
            segments.append(segment)
            segment = []

    residues_to_delete = []
    for segment in segments:
        if len(segment) <= cutoff:
            residues_to_delete += segment

    if len(residues_to_delete) > 0:
        molecule = delete_residues(molecule, residues_to_delete)

    return Merge(molecule.atoms)


def renumber_protein_residues(
        molecule: Union[Universe, AtomGroup], template_sequence: str
) -> Universe:
    """
    Renumber a protein structure according to the provided sequence.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        An MDAnalysis molecule holding a protein structure.
    template_sequence: str
        The amino acid sequence to use for renumbering.

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        An MDAnalysis molecule holding the renumbered protein structure.
    """
    # create a Universe from given molecule
    molecule = Merge(molecule.atoms)

    target_sequence_aligned, template_sequence_aligned = get_structure_sequence_alignment(
        molecule, template_sequence
    )

    target_sequence_counter = 0
    resids = []
    for i, (target_sequence_residue, template_sequence_residue) in enumerate(
            zip(target_sequence_aligned, template_sequence_aligned), start=1
    ):
        if target_sequence_residue != "-":
            resids.append(i)
            target_sequence_counter += 1
    molecule.add_TopologyAttr("resid", resids)

    return molecule


def update_residue_identifiers(
        molecule: Union[Universe, AtomGroup],
        keep_protein_residue_ids: bool = True,
        keep_chain_ids: bool = False,
) -> Universe:
    """
    Update the atom, residue and chain IDs of the given molecular structure. All residues become
    part of chain A, unless 'keep_chain_ids' is set True. Atom IDs will start from 1. Residue IDs
    will start from 1, except 'keep_protein_residue_ids' is set True. This is especially useful, if
    molecules were merged, which can result in overlapping atom and residue IDs as well as
    separate chains.

    Parameters
    ----------
    molecule: MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.Atomgroup
        The MDAnalysis molecule structure for updating atom and residue ids.
    keep_protein_residue_ids: bool
        If the protein residues should be kept.
    keep_chain_ids: bool
        If the chain IDS should be kept.

    Returns
    -------
    : MDAnalysis.core.universe.Universe
        The MDAnalysis molecule structure with updated atom and residue ids.
    """
    # create new Universe
    molecule = Merge(molecule.atoms)
    highest_resid = 1

    # update protein resids
    protein = molecule.select_atoms("protein or resname NMA")
    if len(protein.residues) > 0:
        protein = Merge(protein.atoms)
        if not keep_protein_residue_ids:
            protein_resids = list(range(1, len(protein.residues) + 1))
            protein.add_TopologyAttr("resid", protein_resids)
        highest_resid = protein.residues[-1].resid

    # update resids of non-protein residues except water
    hetero = molecule.select_atoms("not protein and not resname NMA and not resname HOH")
    if len(hetero.residues) > 0:
        hetero = Merge(hetero.atoms)
        hetero_resids = list(range(
            highest_resid + 1, len(hetero.residues) + highest_resid + 1
        ))
        hetero.add_TopologyAttr("resid", hetero_resids)
        highest_resid = hetero.residues[-1].resid

    # update water resids
    water = molecule.select_atoms("resname HOH")
    if len(water.residues) > 0:
        water = Merge(water.atoms)
        water_resids = list(range(
            highest_resid + 1, len(water.residues) + highest_resid + 1
        ))
        water.add_TopologyAttr("resid", water_resids)

    # merge everything into a single Universe
    components = [component for component in [protein, hetero, water] if len(component.atoms) > 0]
    molecule = Merge(components[0].atoms)
    if len(components) > 1:
        for component in components[1:]:
            molecule = Merge(molecule.atoms, component.atoms)

    # update chain ID
    if not keep_chain_ids:
        molecule.add_TopologyAttr("segid", ["A"] * len(molecule.segments))

    return Merge(molecule.atoms)
