from kinoml.core.conditions import AssayConditions
import random
from itertools import product

from ...datasets.core import DatasetProvider as _DatasetProvider
from ...core.ligands import SmilesLigand as _SmilesLigand
from ...core.proteins import UniprotProtein as _UniprotProtein
from ...core.systems import ProteinLigandComplex as _ProteinLigandComplex
from ...core.measurements import pIC50Measurement as _pIC50Measurement


class RandomMeasurementsProteinLigandSystems(_DatasetProvider):
    @classmethod
    def from_source(cls, measurement_type=_pIC50Measurement):
        alkanes = ["C" * i for i in range(1, 10)]
        uniprot_ids = ["P41567", "P8435463"]

        ligands = []
        for alkane in alkanes:
            ligands.append(_SmilesLigand(alkane, metadata={"smiles": alkane}))

        proteins = []
        for uniprot_id in uniprot_ids:
            proteins.append(_UniprotProtein(uniprot_id, metadata={"uniprot_id": uniprot_id}))

        measurements = []
        for ligand, protein in product(ligands, proteins):
            system = _ProteinLigandComplex([protein, ligand])
            conditions = AssayConditions()
            measurement = measurement_type(
                values=[random.uniform(*measurement_type.RANGE)],
                system=system,
                conditions=conditions,
                metadata={"source": "Synthetic dataset"},
            )
            measurements.append(measurement)

        return cls(measurements)
