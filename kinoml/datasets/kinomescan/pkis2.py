from collections import defaultdict
from typing import List, AnyStr, Dict, Any, Callable
from pathlib import Path

import pandas as pd

from .utils import KINOMEScanMapper
from .core import KinomeScanDatasetProvider
from ...core.protein import AminoAcidSequence
from ...core.ligand import Ligand
from ...core.measurements import PercentageDisplacementMeasurement
from ...core.conditions import AssayConditions
from ...features.core import BaseFeaturizer
from ...utils import datapath, defaultdictwithargs


class PKIS2DatasetProvider(KinomeScanDatasetProvider):

    """
    Loads PKIS2 dataset as provided in _Progress towards a public chemogenomic set
    for protein kinases and a call for contributions_[^1].

    [^1]: DOI: 10.1371/journal.pone.0181585

    It will build a dataframe where the SMILES-representation of ligands are the index
    and the columns are the kinase names. To map between KINOMEscan kinase names and
    actual sequences, helper object `kinoml.datatasets.kinomescan.utils.KINOMEScanMapper`
    is instantiated as a class attribute.

    Parameters:
        featurizers: Modify the raw chemical data into other representations.
        raw_datasheet: CSV file to load PKIS2 data from. If the file format is
            different than the default, subclass and reimplement `self._read_dataframe`.
        assay_conditions: Conditions in which the experiment took place. Default is

    __Attributes__

    - `kinases`: Dict that will generate and cache `AminoAcidSequence` objects upon access,
        with keys being any of the KINOMEScan kinase names
    - `ligands`: Dict that will generate and cache `Ligand` objects upon access, with keys
      being any of the available SMILES
    - `available_kinases`: All possible kinase names available in this dataset
    - `available_ligands`: All possible SMILES available in this dataset

    __Examples__

    ```python
    >>> from kinoml.datasets.kinomescan.pkis2 import PKIS2DatasetProvider
    >>> provider = PKIS2DatasetProvider()
    >>> kin = provider.kinases["ABL2"]
    >>> lig = provider.ligands[provider.available_ligands[0]]
    >>> measurement = provider.measurements[kin, lig]
    >>> print(f"% displacement for kinase={kin.header} and ligand={lig.to_smiles()} is {measurement}"
    ```
    """

    def __init__(
        self,
        featurizers: List[BaseFeaturizer] = None,
        raw_spreadsheet: Any[AnyStr, Path] = datapath("kinomescan/journal.pone.0181585.s004.csv"),
        assay_conditions: AssayConditions = AssayConditions(pH=7.0),
        *args,
        **kwargs
    ):
        self.raw_spreadsheet = raw_spreadsheet
        self.assay_conditions = assay_conditions

        self._df = self._read_dataframe(self.raw_spreadsheet)
        self.available_kinases: List[str] = self._df.columns.tolist()
        # TODO: this might be a wrong assumption if SMILES are malformed?
        self.available_ligands: List[str] = self._df.index.tolist()

        # Lazy dicts that will only create objects on key access
        self.kinases = defaultdictwithargs(self._process_kinase)
        self.ligands = defaultdictwithargs(self._process_ligand)
        self.measurements = defaultdictwithargs(self._process_measurement)

        # Featurizers
        self._featurizers = featurizers

    def _read_dataframe(self, filename):
        """
        Consume raw datasheet into a Pandas dataframe. This method must
        provide a Dataframe with the following parameters:

        - The index must be SMILES
        - Column must be kinase names
        - Values are percentage displacement

        """
        # Kinase names are columns 7>413. Smiles appear at column 3.
        return pd.read_csv(filename, usecols=[3] + list(range(7, 413)), index_col=0)

    def _process_kinase(self, name):
        """
        Given the name of a kinase, query NCBI for its sequence. This uses
        `self._kinase_name_mapper`, an instance of `KINOMEScanMapper`.
        """
        sequence = self._kinase_name_mapper.sequence_for_name(name)
        return AminoAcidSequence(sequence, header=name)

    def _process_ligand(self, ligand):
        """
        Helper to build a Ligand from a SMILES string. Result will be cached
        in a per-instance dictionary (see `__init__`).
        """
        if isinstance(ligand, str):
            return Ligand.from_smiles(ligand)
        return ligand

    def _process_measurement(self, kinase_ligand):
        """
        Helper to return the measurement concerning a protein
        and a ligand. Key must be a 2-tuple of (AminoAcidSequence, Ligand).

        This will access the internal dataframe using the provenance information
        (original unprocessed SMILES, Kinase name) to get the number and build
        a PercentageDisplacementMeasure object with the relevant assay conditions.

        Result will be stored in a per-instance dictionary (see `__init__`).
        """
        assert len(kinase_ligand) == 2, "Key must be a 2-tuple of (AminoAcidSequence, Ligand)."
        kinase, ligand = kinase_ligand
        if not isinstance(kinase, AminoAcidSequence):
            raise TypeError("`kinase` must be a kinoml.core.protein.AminoAcidSequence object")
        if not isinstance(ligand, Ligand):
            raise TypeError("`ligand` must be a kinoml.core.ligand.Ligand object")

        smiles = ligand._provenance["smiles"]
        measurement = self._df.loc[smiles, kinase.header]
        return PercentageDisplacementMeasurement(
            measurement, conditions=self.assay_conditions, components=[kinase, ligand],
        )

    def featurize(self):
        return super().featurize()
