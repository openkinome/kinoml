from collections import defaultdict

import pandas as pd

from .utils import KINOMEScanMapper
from ..core import BaseDatasetProvider
from ...core.protein import AminoAcidSequence
from ...core.ligand import Ligand
from ...core.measurements import PercentageDisplacementMeasurement
from ...core.conditions import AssayConditions
from ...utils import datapath, defaultdictwithargs


class PKIS2DatasetProvider:

    """
    Loads PKIS2 dataset as providd in `Progress towards a public chemogenomic set
    for protein kinases and a call for contributions` (DOI: 10.1371/journal.pone.0181585)

    It will build a dataframe where the SMILES-representation of ligands are the index
    and the columns are the kinase names. To map between KINOMEscan kinase names and
    actual sequences, helper object ``kinoml.datatasets.kinomescan.utils.KINOMEScanMapper``
    is instantiated as a class attribute.

    Parameters
    ----------
    featurizers : list of callables, optional=None
        Callables that will modify the raw chemical data into other representations.

    Attributes
    ----------
    kinases : lazy dict, str->AminoAcidSequence
        Dict that will generate and cache AminoAcidSequence objects upon access,
        with keys being any of the KINOMEScan kinase names
    ligands : lazy dict, str->Ligand
        Dict that will generate and cache Ligand objects upon access, with keys
        being any of the available SMILES
    available_kinases : list of str
        All possible kinase names available in this dataset
    available_ligands : list of str
        All possible SMILES available in this dataset

    Class attributes
    ----------------
    _RAW_DATASHEET : str
        CSV file to load PKIS2 data from. If the file format is
        different (columns, etc), sublass and reimplement ``self._read_dataframe``.

    Examples
    --------

    >>> from kinoml.datasets.kinomescan.pkis2 import PKIS2DatasetProvider
    >>> provider = PKIS2DatasetProvider()
    >>> kin = provider.kinases["ABL2"]
    >>> lig = provider.ligands[provider.available_ligands[0]]
    >>> measurement = provider.measurements[kin, lig]
    >>> print(f"% displacement for kinase={kin.header} and ligand={lig.to_smiles()} is {measurement}"

    """

    _RAW_SPREADSHEET = datapath("kinomescan/journal.pone.0181585.s004.csv")
    _kinase_name_mapper = KINOMEScanMapper()

    ASSAY_CONDITIONS = AssayConditions(pH=7.0)

    def __init__(self, featurizers=None, *args, **kwargs):
        self._df = self._read_dataframe(self._RAW_SPREADSHEET)
        self.available_kinases = self._df.columns.tolist()
        self.available_ligands = self._df.index.tolist()

        # Lazy dicts that will only create objects on key accesss
        self.kinases = defaultdictwithargs(self._process_kinase)
        self.ligands = defaultdictwithargs(self._process_ligand)
        self.measurements = defaultdictwithargs(self._process_measurement)

        # Featurizers
        self._featurizers = featurizers

    def _read_dataframe(self, filename):
        # Kinase names are columns 7>413. Smiles appear at column 3.
        return pd.read_csv(filename, usecols=[3] + list(range(7, 413)), index_col=0)

    def _process_kinase(self, name):
        sequence = self._kinase_name_mapper.sequence_for_name(name)
        return AminoAcidSequence(sequence, header=name)

    def _process_ligand(self, ligand):
        if isinstance(ligand, str):
            return Ligand.from_smiles(ligand)
        return ligand

    def _process_measurement(self, kinase_ligand):
        assert len(kinase_ligand) == 2, "key must be (kinase, ligand)"
        kinase, ligand = kinase_ligand
        if not isinstance(kinase, AminoAcidSequence):
            raise TypeError(
                "`kinase` must be a kinoml.core.protein.AminoAcidSequence object"
            )
        if not isinstance(ligand, Ligand):
            raise TypeError("`ligand` must be a kinoml.core.ligand.Ligand object")

        smiles = ligand._provenance["smiles"]
        measurement = self._df.loc[smiles, kinase.header]
        return PercentageDisplacementMeasurement(
            measurement, conditions=self.ASSAY_CONDITIONS, components=[kinase, ligand],
        )

    def featurize(self):
        return super().featurize()
