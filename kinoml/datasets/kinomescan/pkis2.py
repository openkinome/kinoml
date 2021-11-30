from typing import AnyStr, Union
from pathlib import Path
import math

import pandas as pd

from .utils import KINOMEScanMapper
from .core import KinomeScanDatasetProvider
from ...core.proteins import AminoAcidSequence
from ...core.ligands import SmilesLigand
from ...core.systems import ProteinLigandComplex
from ...core.measurements import BaseMeasurement, PercentageDisplacementMeasurement
from ...core.conditions import BaseConditions, AssayConditions
from ...utils import datapath


class PKIS2DatasetProvider(KinomeScanDatasetProvider):

    """
    Loads PKIS2 dataset as provided in _Progress towards a public chemogenomic set
    for protein kinases and a call for contributions [1][].

    [1]: DOI: 10.1371/journal.pone.0181585

    It will build a dataframe where the SMILES-representation of ligands are the index
    and the columns are the kinase names. To map between KINOMEscan kinase names and
    actual sequences, helper object `kinoml.datatasets.kinomescan.utils.KINOMEScanMapper`
    is instantiated as a class attribute.

    Examples
    --------
    >>> from kinoml.datasets.kinomescan.pkis2 import PKIS2DatasetProvider
    >>> provider = PKIS2DatasetProvider.from_source()
    >>> system = provider.systems[0]
    >>> print(f"% displacement for kinase={system.protein.name} and "
    >>>        "ligand={system.ligand.to_smiles()} is {system.measurement}")

    """

    @classmethod
    def from_source(  # pylint: disable=arguments-differ
        cls,
        path_or_url: Union[AnyStr, Path] = datapath("kinomescan/journal.pone.0181585.s004.csv"),
        measurement_type: BaseMeasurement = PercentageDisplacementMeasurement,
        conditions: BaseConditions = AssayConditions(pH=7.0),
    ):
        """
        Create a DatasetProvider out of the raw data in a file

        Parameters
        ----------
        path_or_url : str
            CSV file with the protein-ligand measurements
        measurement_type : BaseMeasurement
            Which type of measurement was taken for each protein-ligand pair
        conditions : BaseConditions
            Experimental conditions of the assay

        Note
        ----
        TODO:

        - Investigate lazy access and object generation
        """
        filename = cls._download_to_cache_or_retrieve(path_or_url)
        df = cls._read_dataframe(filename)
        df = df[df.index.notna()]

        # Read in proteins
        mapper = KINOMEScanMapper()
        kinases = []
        for kin_name in df.columns:
            sequence = mapper.sequence_for_name(kin_name)
            accession = mapper.accession_for_name(kin_name)
            mutations = mapper.mutations_for_name(kin_name)
            if math.isnan(mutations):
                mutations = None
            start_stop = mapper.start_stop_for_name(kin_name)
            metadata = {"accession": accession, "mutations": mutations, "start_stop": start_stop}
            kinases.append(AminoAcidSequence(sequence, name=kin_name, metadata=metadata))

        # Read in ligands
        ligands = []
        for smiles in df.index:
            # We only read the SMILES for now. Promoting to full-fledged objects
            # can be done through featurizers (see SmilesToLigandFeaturizer)
            ligand = SmilesLigand.from_smiles(smiles, name=smiles)
            ligands.append(ligand)

        lol = list(df.itertuples(index=False, name=None))  # FIXME: This might be dangerous

        # Build ProteinLigandComplex objects
        systems = {}
        measurements = []
        for i, ligand in enumerate(ligands):
            for j, kinase in enumerate(kinases):
                value = lol[i][j]
                if not value and value != 0:
                    continue  # this is a nan cell
                key = (ligand.to_smiles(), kinase.sequence)
                if key not in systems:
                    systems[key] = ProteinLigandComplex([kinase, ligand])
                measurement = measurement_type(value, conditions=conditions, system=systems[key])
                measurements.append(measurement)

        return cls(measurements=measurements, metadata={"path_or_url": path_or_url})

    @staticmethod
    def _read_dataframe(filename: Union[AnyStr, Path]) -> pd.DataFrame:
        """
        Consume raw datasheet into a Pandas dataframe. This method must
        provide a Dataframe with the following parameters:

        - The index must be SMILES
        - Column must be kinase names
        - Values are percentage displacement

        """
        # Kinase names are columns 7>413. Smiles appear at column 3.
        return pd.read_csv(filename, usecols=[3] + list(range(7, 413)), index_col=0)
