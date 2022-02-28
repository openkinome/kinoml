import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .core import DatasetProvider
from ..core.proteins import KLIFSKinase
from ..core.ligands import Ligand
from ..core.systems import ProteinLigandComplex
from ..core.measurements import PercentageDisplacementMeasurement
from ..core.conditions import AssayConditions
from ..utils import datapath


logger = logging.getLogger(__name__)


class PKIS2DatasetProvider(DatasetProvider):

    """
    Loads the PKIS2 dataset as provided in _Progress towards a public chemogenomic set for protein
    kinases and a call for contributions [1].

    [1]: DOI: 10.1371/journal.pone.0181585

    Examples
    --------
    >>> from kinoml.datasets.pkis2 import PKIS2DatasetProvider
    >>> provider = PKIS2DatasetProvider.from_source()
    >>> provider
    """

    @classmethod
    def from_source(
        cls,
        path_or_url: Union[str, Path] = datapath("kinomescan/journal.pone.0181585.s004.csv"),
        path_or_url_constructs: Union[str, Path] = datapath(
            "kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv"
        ),
    ):
        """
        Create a PKIS2 DatasetProvider from the raw data.

        Parameters
        ----------
        path_or_url: str or pathlib.Path
            CSV file with the protein-ligand measurements.
        path_or_url_constructs: str or pathlib.Path
            CSV file with the construct information.
        """
        logger.debug("Loading CSV with construct information ...")
        constructs_df = pd.read_csv(path_or_url_constructs)

        logger.debug("Creating protein objects ...")
        kinases = dict()
        for _, construct in constructs_df.iterrows():
            if construct["Construct Description"] != "Wild Type":
                # mutants not in measurements
                continue
            discoverx_id = construct["DiscoverX Gene Symbol"]
            ncbi_id = construct["Accession Number"]
            if construct["AA Start/Stop"] == "Null":
                # ambiguous, will consider full sequence
                kinase = KLIFSKinase(
                    name=discoverx_id,
                    ncbi_id=ncbi_id,
                )
            else:
                first, last = [x[1:] for x in construct["AA Start/Stop"].split("/")]
                kinase = KLIFSKinase(
                    name=discoverx_id,
                    ncbi_id=ncbi_id,
                    metadata={"construct_range": f"{first}-{last}"},
                )
            kinases[discoverx_id] = kinase

        logger.debug("Loading CSV with measurements ...")
        # column 0 is name, column 3 is smiles, column 7 - 412 are measurements for each kinase
        measurements_df = pd.read_csv(path_or_url, usecols=[0, 3] + list(range(7, 413)))

        logger.debug("Creating systems and measurements ...")
        measurements = []
        kinase_names = measurements_df.columns[2:]
        for _, ligand_measurements in measurements_df.iterrows():
            ligand_name = ligand_measurements["Regno"]
            smiles = ligand_measurements["Smiles"]
            if ligand_name == "0":
                ligand_name = smiles
            ligand = Ligand(smiles=smiles, name=ligand_name)
            for kinase_name, inhibition_value in zip(kinase_names, ligand_measurements.values[2:]):
                measurement = PercentageDisplacementMeasurement(
                    inhibition_value,
                    conditions=AssayConditions(pH=7.0),
                    system=ProteinLigandComplex(components=[ligand, kinases[kinase_name]]),
                )
                measurements.append(measurement)

        return cls(measurements=measurements, metadata={"path_or_url": path_or_url})
