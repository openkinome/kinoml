"""
Splitting strategies for datasets
"""
import random


class BaseGrouper:
    def __init__(self):
        pass

    def assign(self, dataset, overwrite=False):
        groups = self._assign(dataset)
        for key, indices in groups.items():
            for index in indices:
                ms = dataset.measurements[index]
                if not overwrite and ms.group is not None:
                    raise ValueError(
                        f"Cannot assign group to `{ms}` because a group is "
                        f"already assigned: {ms.group}. Choose `overwrite=True` "
                        f"to ignore existing groups."
                    )
                dataset.measurements[index].group = key
        return dataset

    def _assign(self, dataset):
        raise NotImplementedError("Implement in your subclass")


class RandomGrouper(BaseGrouper):

    """
    Randomized groups following a split proportional to the provided ratios

    Parameters:
        ratios: tuple or dict
            1-based ratios for the different groups. They must sum 1.0. If a
            dict is provided, the keys are used to label the resulting groups.
            Otherwise, the groups are 0-enumerated.

    """

    def __init__(self, ratios):
        if isinstance(ratios, (list, tuple)):
            ratios = {i: ratio for i, ratio in enumerate(ratios)}
        assert sum(ratios.values()) == 1, f"`ratios` must sum 1, but you provided {ratios}"
        self.ratios = ratios

    def _assign(self, dataset):
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


class BaseFilter(BaseGrouper):
    pass

