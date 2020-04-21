import logging
from typing import Iterable
from copy import deepcopy

import pandas as pd

from ..core.systems import System
from ..features.core import BaseFeaturizer

logger = logging.getLogger(__name__)


class BaseDatasetProvider:

    """
    Base object for all DatasetProvider classes.

    Parameters:
        systems: A DatasetProvider holds a list of `kinoml.core.systems.System` objects
            (or any of its subclasses). A `System` is a collection of `MolecularComponent`
            objects (e.g. protein or ligand-like entities), plus an optional `Measurement`.
    """

    _raw_data = None

    def __init__(
        self, systems: Iterable[System], *args, **kwargs,
    ):
        self.systems = systems

    @classmethod
    def from_source(cls, filename=None, **kwargs):
        """
        Parse CSV/raw file to object model. This method is responsible of generating
        the objects for `self.data` and `self.measurements`, if relevant.
        Additional kwargs will be passed to `__init__`
        """
        raise NotImplementedError

    def featurize(self, *featurizers: Iterable[BaseFeaturizer]) -> System:
        """
        Given a collection of `kinoml.features.core.BaseFeaturizers`, apply them
        to the present systems.

        Parameters:
            featurizers: Featurization schemes that will be applied to the system,
                in a stacked way.

        !!! todo
            * Do we want to support parallel featurizing too or only stacked featurization?
            * Shall we modify the system in place (default now), return the modified copy or store it?
        """
        # Do we assume the dataset is homogeneous (single type of system and associated measurement)?
        # That would allow to only check once (e.g. test support for first system)
        for system in self.systems:
            for featurizer in featurizers:
                featurizer.supports(system, raise_errors=True)
                # .supports() will test for system type, type of components, type of measurement, etc
                system["last"] = system[featurizer.name] = featurizer.featurize(
                    system, inplace=True
                )

    def featurized_data(self):
        for ms in self.systems:
            yield ms.featurizations["last"]

    def _to_dataset(self, style="pytorch"):
        """
        Generate a clean <style>.data.Dataset object for further steps
        in the pipeline (model building, etc).

        !!! Note
            This step is lossy because the resulting objects will no longer
            hold chemical data. Operations depending on such information,
            must be performed first.

        __Examples__

        ```python
        >>> provider = DatasetProvider()
        >>> provider.featurize()  # optional
        >>> splitter = TimeSplitter()
        >>> split_indices = splitter.split(provider.data)
        >>> dataset = provider.to_dataset("pytorch")  # .featurize() under the hood
        >>> X_train, X_test, y_train, y_test = train_test_split(dataset, split_indices)
        ```
        """
        raise NotImplementedError

    def to_dataframe(self, *args, **kwargs):
        """
        Generates a `pandas.DataFrame` containing information on the systems
        and their measurements

        Returns:
            pandas.DataFrame
        """
        if not self.systems:
            return pd.DataFrame()
        s = self.systems[0]
        records = [
            [s.__class__.__name__, "n_components", f"Avg {s.measurement.__class__.__name__}",]
        ]
        for system in self.systems:
            records.append([system.name, len(system.components), system.measurement.values.mean()])
        return pd.DataFrame.from_records(records[1:], columns=records[0])

    def to_pytorch(self, *args, **kwargs):
        raise NotImplementedError

    def to_tensorflow(self, *args, **kwargs):
        raise NotImplementedError

    def to_numpy(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def assay_conditions(self):
        conditions = set()
        for system in self.systems:
            conditions.add(system.measurement.conditions)
        return conditions

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.systems)} systems>"


class ProteinLigandDatasetProvider(BaseDatasetProvider):
    pass
