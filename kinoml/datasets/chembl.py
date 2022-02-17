"""
Creates DatasetProvider objects from ChEMBL activity data
"""
import random

import pandas as pd
from tqdm.auto import tqdm

from .core import MultiDatasetProvider
from ..core.conditions import AssayConditions
from ..core.proteins import KLIFSKinase
from ..core.ligands import Ligand
from ..core.systems import ProteinLigandComplex
from ..core.measurements import pIC50Measurement, pKiMeasurement, pKdMeasurement


class ChEMBLDatasetProvider(MultiDatasetProvider):

    """
    This provider relies heavily on ``openkinome/kinodata`` data ingestion
    pipelines. It will load ChEMBL activities from its Releases page.
    """

    @classmethod
    def from_source(
        cls,
        path_or_url="https://github.com/openkinome/datascripts/releases/download/v0.2/activities-chembl28_v0.2.zip",
        measurement_types=("pIC50", "pKi", "pKd"),
        sample=None,
    ):
        """
        Create a MultiDatasetProvider out of the raw data contained in the zip file

        Parameters
        ----------
        path_or_url : str, optional
            path or URL to a (zipped) CSV file containing activities from ChEMBL,
            using schema detailed below.
        measurement_types : tuple of str, optional
            Which measurement types must be imported from the CSV. By default, all
            three (pIC50, pKi, pKd) will be loaded, but you can choose a subset (
            e.g. ``("pIC50",)``).
        sample : int, optional=None
            If set to larger than zero, load only N data points from the dataset.

        Note
        ----
        ChEMBL aggregates data from lots of sources, so conditions are guaranteed
        to be different across experiments.

        """
        cached_path = cls._download_to_cache_or_retrieve(path_or_url)
        df = pd.read_csv(cached_path)
        df = df.dropna(
            subset=[
                "compound_structures.canonical_smiles",
                "component_sequences.sequence",
                "activities.standard_type",
            ]
        )
        measurement_type_classes = {
            "pIC50": pIC50Measurement,
            "pKi": pKiMeasurement,
            "pKd": pKdMeasurement,
        }
        measurements = []
        systems = {}
        kinases = {}
        ligands = {}
        chosen_types_labels = df["activities.standard_type"].isin(set(measurement_types))
        filtered_records = df[chosen_types_labels].to_dict("records")
        if sample is not None:
            filtered_records = random.sample(filtered_records, sample)
        for row in tqdm(filtered_records):
            try:
                measurement_type_key = row["activities.standard_type"]
                kinase_key = row["component_sequences.sequence"]
                ligand_key = row["compound_structures.canonical_smiles"]
                system_key = (kinase_key, ligand_key)
                if kinase_key not in kinases:
                    metadata = {
                        "uniprot_id": row["UniprotID"],
                        "chembl_target_id": row["target_dictionary.chembl_id"],
                    }
                    kinase = KLIFSKinase(
                        sequence=kinase_key,
                        name=row["UniprotID"],
                        uniprot_id=row["UniprotID"],
                        metadata=metadata
                    )
                    kinases[kinase_key] = kinase
                if ligand_key not in ligands:
                    ligands[ligand_key] = Ligand(smiles=ligand_key, name=ligand_key)
                if system_key not in systems:
                    systems[system_key] = ProteinLigandComplex(
                        [kinases[kinase_key], ligands[ligand_key]]
                    )

                MeasurementType = measurement_type_classes[measurement_type_key]
                conditions = AssayConditions(pH=7)
                system = systems[system_key]
                metadata = {
                    "unit": f"-log10({row['activities.standard_units']}E-9)",
                    "confidence": row["assays.confidence_score"],
                    "chembl_activity": row["activities.activity_id"],
                    "chembl_document": row["docs.chembl_id"],
                    "year": row["docs.year"],
                }
                measurement = MeasurementType(
                    values=row["activities.standard_value"],
                    system=system,
                    conditions=conditions,
                    metadata=metadata,
                )
                measurements.append(measurement)
            except Exception as exc:
                print("Couldn't process record", row)
                print("Exception:", exc)

        return cls(
            measurements,
            metadata={
                "path_or_url": path_or_url,
                "measurement_types": measurement_types,
                "sample": sample,
            },
        )
