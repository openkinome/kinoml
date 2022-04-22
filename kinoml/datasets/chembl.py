"""
Creates DatasetProvider objects from ChEMBL activity data
"""
import logging
import random

import pandas as pd
from tqdm.auto import tqdm

from .core import MultiDatasetProvider
from ..core.conditions import AssayConditions
from ..core.proteins import Protein, KLIFSKinase
from ..core.ligands import Ligand
from ..core.systems import ProteinLigandComplex
from ..core.measurements import pIC50Measurement, pKiMeasurement, pKdMeasurement


logger = logging.getLogger(__name__)


class ChEMBLDatasetProvider(MultiDatasetProvider):

    """
    This provider relies heavily on ``openkinome/kinodata`` data ingestion
    pipelines. It will load ChEMBL activities from its releases page.
    """

    @classmethod
    def from_source(
        cls,
        path_or_url="https://github.com/openkinome/datascripts/releases/download/v0.2/activities-chembl28_v0.2.zip",
        measurement_types=("pIC50", "pKi", "pKd"),
        uniprot_ids=None,
        sample=None,
        protein_type: str = "KLIFSKinase",
        toolkit: str = "OpenEye",
    ):
        """
        Create a MultiDatasetProvider out of the raw data contained in the zip file.

        Parameters
        ----------
        path_or_url: str, optional
            path or URL to a (zipped) CSV file containing activities from ChEMBL,
            using schema detailed below.
        measurement_types: tuple of str, optional
            Which measurement types must be imported from the CSV. By default, all
            three (pIC50, pKi, pKd) will be loaded, but you can choose a subset (
            e.g. ``("pIC50",)``).
        uniprot_ids: None or list of str, default=None
            Restrict measurements to the given UniProt IDs.
        sample: int, optional=None
            If set to larger than zero, load only N data points from the dataset.
        protein_type: str, default=KLIFSKinase
            The protein object type to use ('Protein' or 'KLIFSKinase').
        toolkit: str, default=OpenEye
            The toolkit to use for creating protein objects (e.g. 'OpenEye', 'MDAnalysis'),
            allowed values depend on the specified `protein_type`.

        Raises
        ------
        ValueError
            Given protein_type {protein_type} is not valid, only {allowed_protein_types} are
            allowed.

        Note
        ----
        ChEMBL aggregates data from lots of sources, so conditions are guaranteed
        to be different across experiments.
        """
        logger.debug("Checking protein type ...")
        protein_type_classes = {"Protein": Protein, "KLIFSKinase": KLIFSKinase}
        if protein_type not in protein_type_classes.keys():
            raise ValueError(
                f"Given protein_type {protein_type} is not valid, "
                f"only {protein_type_classes.keys()} are allowed."
            )

        logger.debug("Retrieving and reading CSV ...")
        cached_path = cls._download_to_cache_or_retrieve(path_or_url)
        df = pd.read_csv(cached_path)
        df = df.dropna(
            subset=[
                "compound_structures.canonical_smiles",
                "component_sequences.sequence",
                "activities.standard_type",
            ]
        )

        if uniprot_ids:
            logger.debug(f"Filtering for UniProt IDs {uniprot_ids}...")
            df = df[df["UniprotID"].isin(uniprot_ids)]

        logger.debug(f"Filtering for measurement types {measurement_types} ...")
        chosen_types_labels = df["activities.standard_type"].isin(set(measurement_types))
        filtered_records = df[chosen_types_labels].to_dict("records")

        if sample is not None:
            logger.debug(f"Getting sample of size {sample} ...")
            filtered_records = random.sample(filtered_records, sample)

        measurement_type_classes = {
            "pIC50": pIC50Measurement,
            "pKi": pKiMeasurement,
            "pKd": pKdMeasurement,
        }
        measurements = []
        systems = {}
        proteins = {}
        ligands = {}
        logger.debug(f"Creating systems and measurements ...")
        for row in tqdm(filtered_records):
            try:
                measurement_type_key = row["activities.standard_type"]
                protein_key = row["component_sequences.sequence"]
                ligand_key = row["compound_structures.canonical_smiles"]
                system_key = (protein_key, ligand_key)
                if protein_key not in proteins:
                    metadata = {
                        "uniprot_id": row["UniprotID"],
                        "chembl_target_id": row["target_dictionary.chembl_id"],
                    }
                    protein = protein_type_classes[protein_type](
                        sequence=protein_key,
                        name=row["UniprotID"],
                        uniprot_id=row["UniprotID"],
                        metadata=metadata,
                        toolkit=toolkit,
                    )
                    proteins[protein_key] = protein
                if ligand_key not in ligands:
                    ligands[ligand_key] = Ligand(smiles=ligand_key, name=ligand_key)
                if system_key not in systems:
                    systems[system_key] = ProteinLigandComplex(
                        [proteins[protein_key], ligands[ligand_key]]
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
