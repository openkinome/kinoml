from typing import Union, Iterable

import numpy as np

from .conditions import AssayConditions
from .systems import System


class BaseMeasurement:
    """
    We will have several subclasses depending on the experiment.
    They will also provide loss functions tailored to it.

    Values of the measurement can have more than one replicate. In fact,
    single replicates are considered a specific case of a multi-replicate.

    Parameters:
        values: The numeric measurement(s)
        conditions: Experimental conditions of this measurement
        system: Molecular entities measured, contained in a System object
        strict: Whether to perform sanity checks at initialization.

    !!! todo
        Investigate possible uses for `pint`
    """

    def __init__(
        self,
        values: Union[float, Iterable[float]],
        conditions: AssayConditions,
        system: System,
        errors: Union[float, Iterable[float]] = np.nan,
        strict: bool = True,
        metadata: dict = None,
        **kwargs,
    ):
        self._values = np.reshape(values, (1,))
        self._errors = np.reshape(errors, (1,))
        self.conditions = conditions
        self.system = system
        self.metadata = metadata or {}
        if strict:
            self.sanity_checks()

    @property
    def values(self):
        return self._values

    @property
    def errors(self):
        return self._errors

    @classmethod
    def observation_model(cls, backend="pytorch", log10=False):
        """
        The observation_model function must be defined Measurement type, in the appropriate
        subclass. It dispatches to underlying static methods, suffixed by the
        backend (e.g. `_observation_model_pytorch`, `_observation_model_tensorflow`). These methods are
        _static_, so they do not have access to the class. This is done on purpose
        for composability of modular observation_model functions. The parent DatasetProvider
        classes will request just the function (and not the computed value), and
        will pass the needed variables. The signature is, hence, undefined.

        There are some standardized keyword arguments we use by convention, though:

        - `values`
        - `errors`
        """
        return cls._observation_model(backend=backend, log10=log10)

    @classmethod
    def _observation_model(cls, backend="pytorch", log10=False, type_=None):
        assert backend in ("pytorch", "tensorflow"), f"Backend {backend} is not supported!"
        if log10:
            return getattr(cls, f"_observation_model_log10_{backend}")
        return getattr(cls, f"_observation_model_{backend}")

    @staticmethod
    def _observation_model_pytorch(*args, **kwargs):
        raise NotImplementedError("Implement in your subclass!")

    @staticmethod
    def _observation_model_log10_pytorch(*args, **kwargs):
        raise NotImplementedError("Implement in your subclass!")

    def sanity_checks(self):
        """
        Perform some checks for valid values
        """

    def __eq__(self, other):
        return (
            (self.values == other.values).all()
            and self.conditions == other.conditions
            and self.system == other.system
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} values={self.values} conditions={self.conditions!r} system={self.system!r}>"


class PercentageDisplacementMeasurement(BaseMeasurement):

    r"""
    Measurement where the value(s) must be percentage(s) of displacement.

    For the percent displacement measurements available from KinomeScan, we make the assumption (see JDC's notes) that

    $$
    D([I]) \approx \frac{1}{1 + \frac{K_d}{[I]}}
    $$

    For KinomeSCAN assays, all assays are usually performed at a single molar concentration, $ [I] \sim 1 \mu M $.

    We therefore define the following function:

    $$
    \mathbf{F}_{KinomeScan}(\Delta g, [I]) = \frac{1}{1 + \frac{exp[\Delta g] * 1[M]}{[I]}}.
    $$
    """

    def sanity_checks(self):
        super().sanity_checks()
        assert (0 <= self.values <= 100).all(), "One or more values are not in [0, 100]"

    @staticmethod
    def _observation_model_pytorch(values, inhibitor_conc=1e-6, **kwargs):
        import torch

        return 100 / (1 + torch.exp(values) / inhibitor_conc)

    @staticmethod
    def _observation_model_log10_pytorch(*args, **kwargs):
        import torch

        return -torch.log10(
            PercentageDisplacementMeasurement._observation_model_pytorch(*args, **kwargs)
        )


class IC50Measurement(BaseMeasurement):

    r"""
    Measurement where the value(s) come from IC50 experiments

    We use the Cheng Prusoff equation here.

    The [Cheng Prusoff](https://en.wikipedia.org/wiki/IC50#Cheng_Prusoff_equation) equation states the following relationship

    \begin{equation}
    K_i = \frac{IC50}{1+\frac{[S]}{K_m}}
    \end{equation}

    We make the following assumptions here
    1. $[S] = K_m$
    2. $K_i \approx K_d$

    In the future, we will relax these assumptions.

    Under these assumptions, the Cheng-Prusoff equation becomes
    $$
    IC50 \approx 2 * K_d
    $$

    We define the following function
    $$
    \mathbf{F}_{IC50}(\Delta g) = 2 * \mathbf{F}_{K_d}(\Delta g) = 2 * exp[-\Delta g] * 1[M]
    $$
    """

    @staticmethod
    def _observation_model_pytorch(
        values, substrate_conc=1e-6, michaelis_constant=1, inhibitor_conc=1e-6, **kwargs
    ):
        import torch

        return (1 + substrate_conc / michaelis_constant) * torch.exp(values) * inhibitor_conc

    @staticmethod
    def _observation_model_log10_pytorch(*args, **kwargs):
        import torch

        return -torch.log10(IC50Measurement._observation_model_pytorch(*args, **kwargs))


class KiMeasurement(BaseMeasurement):

    r"""
    Measurement where the value(s) come from K_i_ experiments

    We make the assumption that $K_i \approx K_d$ and therefore $\mathbf{F}_{K_i} = \mathbf{F}_{K_d}$.
    """

    @staticmethod
    def _observation_model_pytorch(values, inhibitor_conc=1e-6, **kwargs):
        import torch

        return torch.exp(values) * inhibitor_conc

    @staticmethod
    def _observation_model_log10_pytorch(*args, **kwargs):
        import torch

        return -torch.log10(KiMeasurement._observation_model_pytorch(*args, **kwargs))


class KdMeasurement(BaseMeasurement):

    r"""
    Measurement where the value(s) come from Kd experiments

    We define the following physics-based function
    $$
    \mathbf{F}_{K_d}(\Delta g) = exp[-\Delta g] * 1[M].
    $$

    If we have measurements at different concentrations $I$ (unit [M]) , then the function can further be defined as

    $$
    \mathbf{F}_{K_d}(\Delta g, I) = exp[-\Delta g] * I[M].
        $$
    """

    @staticmethod
    def _observation_model_pytorch(values, inhibitor_conc=1e-6, **kwargs):
        import torch

        return torch.exp(values) * inhibitor_conc

    @staticmethod
    def _observation_model_log10_pytorch(*args, **kwargs):
        import torch

        return -torch.log10(KdMeasurement._observation_model_pytorch(*args, **kwargs))


def null_observation_model(arg):
    return arg
