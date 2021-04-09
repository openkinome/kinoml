"""
Splitting strategies for datasets
"""
import random
from collections import defaultdict

from tqdm.auto import tqdm


class BaseGrouper:
    """
    Base class to assign groups to measurements in a DatasetProvider
    """

    def __init__(self):
        pass

    def assign(self, dataset, overwrite=False, **kwargs):
        """
        Given a DatasetProvider, assign a key to the elements
        of each group, as provided by ``.indices()``

        Parameters
        ----------
        dataset : DatasetProvider
        overwrite : bool, optional=False
            If a measurement has been assigned a group already,
            do not overwrite unless this option is set to True.

        Returns
        -------
        dataset : DatasetProvider
            The same dataset passed in the input, with
            measurements modified in place.
        """
        groups = self.indices(dataset, **kwargs)
        measurements = dataset.measurements
        for key, indices in groups.items():
            for index in indices:
                ms = measurements[index]
                if not overwrite and ms.group is not None:
                    raise ValueError(
                        f"Cannot assign group to `{ms}` because a group is "
                        f"already assigned: {ms.group}. Choose `overwrite=True` "
                        f"to ignore existing groups."
                    )
                ms.group = key
        return dataset

    def indices(self, dataset, **kwargs):
        """
        Given a dataset, create a dictionary that maps keys or labels
        to a set of numerical indices. The strategy to follow will
        depend on the subclass.

        Parameters
        ----------
        dataset : DatasetProvider

        Returns
        -------
        dict
            Maps ``int` or ``str`` to a list of ``int``
        """
        raise NotImplementedError("Implement in your subclass")


class RandomGrouper(BaseGrouper):

    """
    Randomized groups following a split proportional to the provided ratios

    Parameters
    ----------
    ratios : tuple or dict
        1-based ratios for the different groups. They must sum 1.0. If a
        dict is provided, the keys are used to label the resulting groups.
        Otherwise, the groups are 0-enumerated.

    """

    def __init__(self, ratios):
        if isinstance(ratios, (list, tuple)):
            ratios = {i: ratio for i, ratio in enumerate(ratios)}
        assert sum(ratios.values()) == 1, f"`ratios` must sum 1, but you provided {ratios}"
        self.ratios = ratios

    def indices(self, dataset, **kwargs):
        length = len(dataset)
        indices = list(range(length))
        random.shuffle(indices)
        groups = {}
        start = 0
        for key, ratio in self.ratios.items():
            end = start + int(round(ratio * length, 0))
            groups[key] = indices[start:end]
            start = end
        return groups


class CallableGrouper(BaseGrouper):
    """
    A grouper that applies a user-provided function to each Measurement
    in the Dataset. Returned value should be the name of the group.

    Parameters
    ----------
    function : callable
        This function must be able to take a ``Measurement`` object
        and return a ``str`` or ``int``.
    """

    def __init__(self, function):
        self.function = function

    def indices(self, dataset, progress=True):
        iterator = enumerate(dataset.measurements)
        if progress:
            iterator = tqdm(iterator)

        groups = defaultdict(list)
        for i, measurement in iterator:
            key = self.function(measurement)
            groups[key].append(i)
        return groups


class BaseFilter(BaseGrouper):
    pass
