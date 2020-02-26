import abc
import logging

import numpy as np


logger = logging.getLogger(__name__)


class BaseDatasetProvider(abc.ABC):

    """
    Base object for all Dataset classes
    
    Parameters
    ----------
    chemical_data : list of kinoml.core.MolecularSystem-like
    measurements : array-like of shape (len(chemical_data), N)
        We will reshape it for you if you provide shape=(len(chemical_data),)
    featurizers : list of kinoml.features.BaseFeaturizer-like
    
    Attributes
    ----------
    data : list of kinoml.core.MeasuredMolecularSystem-like
    """

    _raw_data = None

    def __init__(
        self, chemical_data, measurements=None, featurizers=None, *args, **kwargs
    ):
        self._data = chemical_data
        if measurements is None:
            measurements = np.empty((len(self.data), 1))
            measurements[:] = np.NaN
        self._measurements = np.reshape(measurements, (len(self.data), -1))
        self.data = [
            MeasuredMolecularSystem(system, measurement)
            for system, measurement in zip(self._data, self._measurements)
        ]
        self.featurizers = featurizers

    @abc.abstractclassmethod
    def from_source(cls, filename=None, **kwargs):
        """
        Parse CSV to object model. This method is responsible of generating
        the objects for `self.data` and `self.measurements`, if relevant.
        Additional kwargs will be passed to `__init__`
        """
        ...

    @abc.abstractmethod
    def featurize(self):
        """
        Apply featurizers to self.data and self.measurements.
        """
        ...

    @abc.abstractproperty
    def featurized_data(self):
        for ms in self.data:
            yield ms.featurized_data
        # This might create a sklearn-compatible view into the features as default

    @abc.abstractmethod
    def to_dataset(
        self, style="pytorch",
    ):
        """
        Generate a clean <style>.data.Dataset object for further steps
        in the pipeline (model building, etc). 
        
        Notes
        -----
        This step is lossy because the resulting objects will no longer
        hold chemical data. Operations depending on such information, 
        must be performed first.
        
        Examples
        --------
        
        >>> provider = DatasetProvider()
        >>> provider.featurize()  # optional
        >>> splitter = TimeSplitter()
        >>> split_indices = splitter.split(provider.data)
        >>> dataset = provider.to_dataset("pytorch")  # .featurize() under the hood
        >>> X_train, X_test, y_train, y_test = train_test_split(dataset, split_indices)
        
        
        """
        ...

    @abc.abstractmethod
    def _to_pytorch(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def _to_tensorflow(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def _to_numpy(self, *args, **kwargs):
        ...


...


provider = PKIS2Provider()
pt_dataset = provider.to_dataset("pytorch", pre_process=None)

