"""
Creates DatasetProvider objects from ChEMBL activity data
"""
from urllib.request import urlopen
import shutil
import random
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .core import MultiDatasetProvider, ProteinLigandDatasetProvider
from ..core.conditions import AssayConditions
from ..core.proteins import AminoAcidSequence
from ..core.ligands import SmilesLigand
from ..core.systems import ProteinLigandComplex
from ..core.measurements import IC50Measurement, KiMeasurement, KdMeasurement
from ..utils import APPDIR


class ChEMBLDatasetProvider(MultiDatasetProvider):

    """
    This provider relies heavily on `openkinome/datascripts` data ingestion
    pipelines. It will load ChEMBL activities from its Releases page
    """

    @classmethod
    def from_source(
        cls,
        filename="https://github.com/openkinome/datascripts/releases/download/v0.1/activities-chembl27.zip",
        measurement_types=("IC50", "Ki", "Kd"),
        sample=None,
        **kwargs
    ):
        """
        Create a MultiDatasetProvider out of the raw data contained in the zip file

        Parameters:
            filename: URL to a zipped CSV file containing activities from ChEMBL,
                using schema detailed below.

        !!! note
            - ChEMBL aggregates data from lots of sources, so conditions are guaranteed
              to be different across experiments.

        !!! todo
            - Versioning for different openkinome/datascripts releases
        """
        csv_filename = "activities-chembl27.csv"
        cached_path = Path(APPDIR.user_cache_dir) / "chembl" / csv_filename
        if not cached_path.is_file():
            # Download zipped CSV and load it with pandas
            with urlopen(filename) as response, TemporaryDirectory() as tmpdir:
                tmpzip = Path(tmpdir) / "chembl.zip"
                with open(tmpzip, "wb") as f:
                    shutil.copyfileobj(response, f)
                with ZipFile(tmpzip) as zf:
                    zf.extractall(tmpdir)
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(tmpdir) / csv_filename, cached_path)
        df = pd.read_csv(cached_path).dropna(subset=["compound_structures.canonical_smiles"])
        # return df

        measurement_type_classes = {
            "IC50": IC50Measurement,
            "Ki": KiMeasurement,
            "Kd": KdMeasurement,
        }
        measurements_by_type = {"IC50": [], "Ki": [], "Kd": []}
        systems = {}
        kinases = {}
        ligands = {}
        filtered_records = df[df["activities.standard_type"].isin(set(measurement_types))].to_dict(
            "records"
        )
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
                        "uniprot": row["UniprotID"],
                        "chembl_target": row["target_dictionary.chembl_id"],
                    }
                    kinase = AminoAcidSequence(
                        kinase_key, name=row["UniprotID"], metadata=metadata
                    )
                    kinases[kinase_key] = kinase
                if ligand_key not in ligands:
                    metadata = {"chembl_compound": None}  # TODO
                    ligands[ligand_key] = SmilesLigand(ligand_key, name=ligand_key)
                if system_key not in systems:
                    systems[system_key] = ProteinLigandComplex(
                        [kinases[kinase_key], ligands[ligand_key]]
                    )

                MeasurementType = measurement_type_classes[measurement_type_key]
                conditions = AssayConditions(pH=7)
                system = ProteinLigandComplex([kinases[kinase_key], ligands[ligand_key]])
                metadata = {
                    "unit": row["activities.standard_units"],
                    "confidence": row["assays.confidence_score"],
                    "chembl_activity": row["activities.activity_id"],
                    "chembl_document": row["docs.doc_id"],
                    "year": row["docs.year"],
                }
                measurement = MeasurementType(
                    values=row["activities.standard_value"],
                    system=systems[system_key],
                    conditions=conditions,
                    metadata=metadata,
                )
                measurements_by_type[measurement_type_key].append(measurement)
            except Exception as exc:
                print("Couldn't process record", row)
                print("Exception", exc)
        providers = [
            _SingleTypeChEMBLDatasetProvider(list(ms))
            for ms in measurements_by_type.values()
            if len(ms)
        ]
        return cls(providers)


class _SingleTypeChEMBLDatasetProvider(ProteinLigandDatasetProvider):
    pass
