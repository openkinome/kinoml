import logging
from typing import Iterable
from copy import deepcopy
from functools import wraps
from operator import attrgetter

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.measurements import BaseMeasurement
from ..features.core import BaseFeaturizer

logger = logging.getLogger(__name__)


class BaseDatasetProvider:

    """
    Base object for all DatasetProvider classes.

    Parameters:
        measurements: A DatasetProvider holds a list of `kinoml.core.measurements.BaseMeasurement`
            objects (or any of its subclasses). They must be of the same type!
    """

    _raw_data = None

    def __init__(
        self, measurements: Iterable[BaseMeasurement], *args, **kwargs,
    ):
        types = {type(measurement) for measurement in measurements}
        assert (
            len(types) == 1
        ), f"Dataset providers can only allow one type of measurement! You provided: {types}"
        self.measurements = measurements

    @classmethod
    def from_source(cls, filename=None, **kwargs):
        """
        Parse CSV/raw file to object model. This method is responsible of generating
        the objects for `self.data` and `self.measurements`, if relevant.
        Additional kwargs will be passed to `__init__`
        """
        raise NotImplementedError

    def featurize(self, *featurizers: Iterable[BaseFeaturizer]):
        """
        Given a collection of `kinoml.features.core.BaseFeaturizers`, apply them
        to the systems present in the `self.measurements`.

        Parameters:
            featurizers: Featurization schemes that will be applied to the systems,
                in a stacked way.

        !!! todo
            * Do we want to support parallel featurizing too or only stacked featurization?
            * Shall we modify the system in place (default now), return the modified copy or store it?
        """
        systems = self.systems
        for featurizer in featurizers:
            # .supports() will test for system type, type of components, type of measurement, etc
            featurizer.supports(next(iter(systems)), raise_errors=True)

        for system in tqdm(systems, desc="Featurizing systems..."):
            for featurizer in featurizers:
                system.featurizations[featurizer.name] = featurizer.featurize(system, inplace=True)
            system.featurizations["last"] = system.featurizations[featurizers[-1].name]

    def clear_featurizations(self):
        for system in self.systems:
            system.featurizations.clear()

    def featurized_systems(self, key="last"):
        return [ms.system.featurizations[key] for ms in self.measurements]

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
        columns = ["Systems", "n_components", self.measurements[0].__class__.__name__]
        records = [
            (
                measurement.system.name,
                len(measurement.system.components),
                measurement.values.mean(),
            )
            for measurement in self.measurements
        ]

        return pd.DataFrame.from_records(records, columns=columns)

    def to_pytorch(self, **kwargs):
        from .torch_datasets import TorchDataset

        dataset = TorchDataset(self.featurized_systems(), self.measurements_as_array(**kwargs))
        return dataset

    def to_tensorflow(self, *args, **kwargs):
        raise NotImplementedError

    def to_numpy(self, *args, **kwargs):
        raise NotImplementedError

    def mapping(self, backend="pytorch"):
        """
        Draft implementation of a modular mapping function, based on individual contributions
        from different measurement types.
        """
        assert backend in ("pytorch",), f"Backend {backend} is not supported!"
        return getattr(self, f"_mapping_{backend}")(self.measurement_type.mapping(backend=backend))

    @staticmethod
    def _mapping_pytorch(mapping):
        def inner_mapping(values, **kwargs):
            """
            Pytorch sum of the tensors returned by the underlying `mapping` function,
            implemented by the `MeasurementType` class. `kwargs` are forwarded blindly.
            Check `self.measurement_type` to explore its `.mapping()` method
            for more information.
            """
            import torch

            tensor = torch.empty(len(values))
            tensor[:] = mapping(values, **kwargs)
            # TODO: Do we want to sum the mappings here or just the losses?
            #       Probably the losses, right?
            return tensor  # .sum(1)

        return inner_mapping

    @staticmethod
    def _loss_tensorflow(**kwargs):
        raise NotImplementedError("Implement in your subclass!")

    @property
    def systems(self):
        return list({ms.system for ms in self.measurements})

    @property
    def measurement_type(self):
        return type(self.measurements[0])

    def measurements_as_array(self, reduce=np.mean):
        import numpy as np

        result = np.empty(len(self.measurements))
        for i, measurement in enumerate(self.measurements):
            result[i] = reduce(measurement.values)
        return result

    @property
    def conditions(self):
        return {ms.conditions for ms in self.measurements}

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"{len(self.measurements)} {self.measurement_type.__name__} measurements "
            f"and {len(self.systems)} systems>"
        )


class ProteinLigandDatasetProvider(BaseDatasetProvider):
    pass
