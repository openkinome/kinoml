import logging
from typing import Iterable

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
        featurizers: Pipeline of descriptor featurization schemes

    """

    _raw_data = None

    def __init__(
        self,
        systems: Iterable[System],
        featurizers: Iterable[BaseFeaturizer] = None,
        *args,
        **kwargs
    ):
        self.systems = systems
        self.featurizers = featurizers

    @classmethod
    def from_source(cls, filename=None, **kwargs):
        """
        Parse CSV/raw file to object model. This method is responsible of generating
        the objects for `self.data` and `self.measurements`, if relevant.
        Additional kwargs will be passed to `__init__`
        """
        raise NotImplementedError

    def featurize(self):
        """
        Apply featurizers to self.data and self.measurements.
        """
        raise NotImplementedError

    def featurized_data(self):
        for ms in self.systems:
            yield ms.featurized_data
        # This might create a sklearn-compatible view into the features as default

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


class ProteinLigandDatasetProvider(BaseDatasetProvider):
    pass
