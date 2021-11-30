"""
Measurements are the first-level citizens in a dataset.

A ``kinoml.datasets.DatasetProvider`` can be essentially considered
a list of ``Measurement`` objects. These objects contain:

- One or more numeric values, stored as an array under the ``.values`` attribute.
- The set of ``MolecularComponent`` objects that were measured, collected
  under a ``System`` in the ``.system`` attribute.
- The conditions the measurement was taken under.

Read on the subclasses for more concrete information about observation models,
loss functions, errors and other features.
"""

from typing import Union, Iterable

import numpy as np

from .conditions import AssayConditions
from .systems import System


LN10 = np.log(10)


class BaseMeasurement:
    """
    We will have several subclasses depending on the experiment.
    They will also provide observation models tailored to it.

    Values of the measurement can have more than one replicate. In fact,
    single replicates are considered a specific case of a multi-replicate.

    Parameters
    ----------
    values : float or array-like of floats
        The numeric measurement(s). If float, it will be
        reshaped to a single-element array.
    errors : float or array-like of floats, optional
        The associated errors to ``values``. Must be same
        shape as ``values``. If float, it will be
        reshaped to a single-element array.
    conditions : AssayConditions
        Experimental conditions of this measurement
    system : System
        Molecular entities measured, contained in a System object
    group : int or str, optional
        A label that identifies this measurement as part of a group.
        Useful to split datasets according to shared properties,
        like research group, measured molecule(s), etc.
    metadata : dict, optional
        Provenance data for this measurement
    strict : bool, optional=True
        Whether to perform safe checks at initialization.

    Attributes
    ----------
    RANGE : tuple of float
        Acceptable range of measurement values, as stored in ``values``

    Note
    ----
    TODO: Investigate possible uses for ``pint``
    """

    RANGE = (-np.inf, np.inf)

    def __init__(
        self,
        values: Union[float, Iterable[float]],
        conditions: AssayConditions,
        system: System,
        errors: Union[float, Iterable[float]] = np.nan,
        group: Union[int, str] = None,
        strict: bool = True,
        metadata: dict = None,
        **kwargs,
    ):
        self._values = np.reshape(values, (1,))
        self._errors = np.reshape(errors, (1,))
        self.conditions = conditions
        self.system = system
        self.group = group
        self.metadata = metadata or {}
        if strict:
            self.check()

    @property
    def values(self):
        return self._values

    @property
    def errors(self):
        return self._errors

    def check(self):
        """
        Perform some checks for valid values
        """
        assert self._values.shape == self._errors.shape, (
            f"Values and errors must match in shape, "
            f"but you provided {self._values.shape} and "
            f"{self._errors.shape}, respectively",
        )

    def __eq__(self, other):
        return (
            (self.values == other.values).all()
            and self.conditions == other.conditions
            and self.system == other.system
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} values={self.values} conditions={self.conditions!r} system={self.system!r}>"


class ObservationModelMeasurement(BaseMeasurement):
    """
    A base class that implements the concept of *observation models* and
    *loss function* adapters.

    Read on the ``.observation_model`` and ``.loss_adapter``
    class methods for more information.
    """

    @classmethod
    def observation_model(cls, backend="pytorch"):
        """
        The observation_model function must be defined per Measurement type, in the appropriate
        subclass. It dispatches to underlying static methods, suffixed by the
        backend (e.g. `_observation_model_pytorch`, `_observation_model_tensorflow`). These methods are
        _static_, so they do not have access to the class. This is done on purpose
        for composability of modular observation_model functions.
        The signature is, hence, undefined.

        There are some standardized keyword arguments we use by convention, though:

        - `values`
        - `errors`
        """
        return cls._observation_model(backend=backend)

    @classmethod
    def _observation_model(cls, backend="pytorch", type_=None):
        try:
            method = getattr(cls, f"_observation_model_{backend}")
        except AttributeError:
            msg = f"Observation model for backend `{backend}` is not available for `{cls}` types"
            raise NotImplementedError(msg)
        else:
            return method

    @staticmethod
    def _observation_model_null(dG_over_KT):
        return dG_over_KT

    @staticmethod
    def _observation_model_pytorch(*args, **kwargs):
        raise NotImplementedError("Implement in your subclass!")

    @staticmethod
    def _observation_model_xgboost(*args, **kwargs):
        raise NotImplementedError("Implement in your subclass!")

    @classmethod
    def loss_adapter(cls, backend="pytorch"):
        """
        Some frameworks require objective functions to include the
        observation model transformation in the same callable. This
        method provides a factory of such methods.
        """
        try:
            return getattr(cls, f"_loss_adapter_{backend}")
        except AttributeError:
            msg = f"Adapter for backend `{backend}` is not available for `{cls}` types"
            raise NotImplementedError(msg)

    @staticmethod
    def _loss_adapter_generic(
        predicted,
        observed,
        loss_func,
        loss_kwargs=None,
        pre_loss_func=None,
        pre_loss_kwargs=None,
        post_loss_func=None,
        post_loss_kwargs=None,
    ):
        if pre_loss_func is not None:
            pre_loss_kwargs = pre_loss_kwargs or {}
            predicted = pre_loss_func(predicted, **pre_loss_kwargs)

        loss_kwargs = loss_kwargs or {}
        loss = loss_func(predicted, observed, **loss_kwargs)

        if post_loss_func is not None:
            post_loss_kwargs = post_loss_kwargs or {}
            loss = post_loss_func(loss, **post_loss_kwargs)

        return loss

    @classmethod
    def _loss_adapter_pytorch(cls, predicted, observed, loss_func, **kwargs):
        kwargs["pre_loss_func"] = cls._observation_model_pytorch
        kwargs["post_loss_func"] = cls._post_loss_adapter
        return cls._loss_adapter_generic(
            predicted=predicted,
            observed=observed,
            loss_func=loss_func,
            **kwargs,
        )

    @staticmethod
    def _post_loss_adapter(loss, **kwargs):
        return loss


class PercentageDisplacementMeasurement(ObservationModelMeasurement):

    r"""
    Measurement where the value(s) must be percentage(s) of displacement.

    For the percent displacement measurements available from KinomeScan, we have the following:

    $$
    D([I]) = \frac{1}{1 + \frac{K_d}{[I]}}
    $$

    We therefore define the following function:

    $$
    \mathbf{F}_{KinomeScan}(\Delta g, [I]) = 100 * \frac{1}{1 + \frac{exp[\Delta g] * C[M]}{[I]}},
    $$

    where $C$ is the standard concentration of 1 [M].

    Note
    ----
    The acceptable range for this measurement is [0, 100], inclusive.
    """
    RANGE = (0, 100)

    def check(self):
        super().check()
        assert (
            self.RANGE[0] <= self.values <= self.RANGE[1]
        ).all(), "One or more values are not in [0, 100]"

    @staticmethod
    def _observation_model_pytorch(dG_over_KT, inhibitor_conc=1, standard_conc=1, **kwargs):
        import torch

        return (100 * inhibitor_conc) / (inhibitor_conc + (standard_conc * torch.exp(dG_over_KT)))
        # return 100 * (1 / (1 + (torch.exp(dG_over_KT) * standard_conc) / inhibitor_conc))

    @staticmethod
    def _observation_model_numpy(dG_over_KT, inhibitor_conc=1, standard_conc=1, **kwargs):
        r"""
        Return the observation model.

        $$
        F(\Delta g) = 100 * \frac{1}{1 + \frac{exp[\Delta g] * C[M]}{[I]}},
        $$
        """
        # TODO: Review the performance penalty of type casting
        dG_over_KT = dG_over_KT.astype("float64")
        return (100 * inhibitor_conc) / (inhibitor_conc + (standard_conc * np.exp(dG_over_KT)))
        # return 100 * 1 / (1 + (np.exp(dG_over_KT) * standard_conc) / inhibitor_conc)

    _observation_model_xgboost = _observation_model_numpy

    @staticmethod
    def _post_loss_adapter(loss, **kwargs):
        #  TODO: Revisit which kind of weighting is needed here
        return loss

    @staticmethod
    def _loss_adapter_xgboost_mse(labels, dG_over_KT, inhibitor_conc=1, standard_conc=1, **kwargs):
        r"""
        Return the gradient and the hessian of the loss defined by

        $$
        L(y, \hat y) = \frac{1}{2} * (y - F(\hat y)) ** 2.
        $$

        See theory notes for more details.
        """
        # TODO: Review the performance penalty of type casting (needed to prevent overflows)
        labels = labels.astype("float64")
        dG_over_KT = dG_over_KT.astype("float64")

        constant = -1 * 100 * inhibitor_conc
        # FIXME: Check these overflows for non-physical calcs
        temp = standard_conc * np.exp(dG_over_KT)
        summation = inhibitor_conc + temp
        difference = 100 * inhibitor_conc / summation - labels

        grad_loss = constant * difference * temp / (summation ** 2)

        first_term = constant * temp / (summation ** 2)
        numerator = temp * summation - 2 * (temp ** 2)

        hess_loss = first_term ** 2 + difference * constant * numerator / (summation ** 3)

        # XGBoost works only with f32
        return grad_loss.astype("float32"), hess_loss.astype("float32")


class pIC50Measurement(ObservationModelMeasurement):

    r"""
    Measurement where the value(s) come from pIC50 experiments

    We use the Cheng Prusoff equation here.

    The `Cheng Prusoff <https://en.wikipedia.org/wiki/IC50#Cheng_Prusoff_equation>`_
    equation states the following relationship:

    $$
    K_i = \frac{IC50}{1+\frac{[S]}{K_m}}
    $$

    We make the following assumption (which will be relaxed in the future):

    $K_i \approx K_d$

    Under this assumptions, the Cheng-Prusoff equation becomes:

    $$
    IC50 \approx {1+\frac{[S]}{K_m}} * K_d
    $$

    We define the following function:

    $$
    \mathbf{F}_{IC_{50}}(\Delta g) = \Big({1+\frac{[S]}{K_m}}\Big) * \mathbf{F}_{K_d}(\Delta g) = \Big({1+\frac{[S]}{K_m}}\Big) * exp[\Delta g] * C[M].
    $$

    Given IC50 values given in molar units, we obtain pIC50
    values in molar units using the tranformation:

    $$
    pIC50 [M] = -log_{10}(IC50[M])
    $$

    Finally the observation model for pIC50 values is:

    $$
    \mathbf{F}_{pIC_{50}}(\Delta g) = - \frac{\Delta g + \ln\Big(\big(1+\frac{[S]}{K_m}\big)*C\Big)}{\ln(10)}.
    $$

    Note
    ----
    The acceptable range for this measurement is [0, 15], inclusive.
    """
    RANGE = (0, 15)

    @staticmethod
    def _observation_model_pytorch(
        dG_over_KT, substrate_conc=1e-6, michaelis_constant=1, standard_conc=1, **kwargs
    ):
        constant = np.log((1 + substrate_conc / michaelis_constant) * standard_conc)
        return -(dG_over_KT + constant) / LN10

    # implementation does not rely on any torch.* methods so we can just reuse it
    # for other backends via aliases
    _observation_model_numpy = _observation_model_pytorch
    _observation_model_xgboost = _observation_model_pytorch

    @staticmethod
    def _loss_adapter_xgboost_mse(
        labels,
        dG_over_KT,
        substrate_conc=1e-6,
        michaelis_constant=1,
        standard_conc=1,
        **kwargs,
    ):
        """
        In XGBoost, observation models need to be applied within the loss function. In this specific case,
        MSE is applied and differentiated (twice) to provide the gradients and hessian matrices.

        $$
        loss = 1/2 * (observation_pIC50(preds)-labels)^2
        $$

        Parameters:
            dmatrix : xgboost.DMatrix
                Passed automatically by the xgboost loop

        """
        constant = np.log((1 + substrate_conc / michaelis_constant) * standard_conc) / LN10

        grad_loss = (labels + (dG_over_KT + constant) / LN10) / LN10
        hess_loss = np.full(grad_loss.shape, 1 / (LN10 * LN10))

        return grad_loss.astype("float32"), hess_loss.astype("float32")

    @staticmethod
    def _post_loss_adapter(loss, **kwargs):
        #  TODO: Revisit which kind of weighting is needed here
        return loss

    def check(self):
        super().check()
        msg = f"Values for {self.__class__.__name__} are expected to be in the [0, 15] range."
        assert (self.RANGE[0] <= self.values <= self.RANGE[1]).all(), msg


class pKiMeasurement(ObservationModelMeasurement):

    r"""
    Measurement where the value(s) come from $K_i$ experiments

    We make the assumption that $K_i \approx K_d$ and therefore

    $\mathbf{F}_{pK_i} = \mathbf{F}_{pK_d}$.

    Note
    ----
    The acceptable range for this measurement is [0, 100], inclusive.

    """

    RANGE = (0, 15)

    @staticmethod
    def _observation_model_pytorch(dG_over_KT, standard_conc=1, **kwargs):
        return -(dG_over_KT + np.log(standard_conc)) / LN10

    # implementation does not rely on any torch.* methods so we can just reuse it
    # for other backends via aliases
    _observation_model_numpy = _observation_model_pytorch
    _observation_model_xgboost = _observation_model_pytorch

    @staticmethod
    def _loss_adapter_xgboost_mse(labels, dG_over_KT, standard_conc=1, **kwargs):
        r"""
        Return the gradient and the hessian of the loss defined by

        $$
        L(y, \hat y) = \frac{1}{2} * (y - F(\hat y)) ** 2.
        $$
        """
        grad_loss = (labels + (dG_over_KT + standard_conc) / LN10) / LN10
        hess_loss = np.full(grad_loss.shape, 1 / (LN10 * LN10))

        return grad_loss.astype("float32"), hess_loss.astype("float32")

    @staticmethod
    def _post_loss_adapter(loss, **kwargs):
        #  TODO: Revisit which kind of weighting is needed here
        return loss

    def check(self):
        super().check()
        msg = f"Values for {self.__class__.__name__} are expected to be in the [0, 15] range."
        assert (self.RANGE[0] <= self.values <= self.RANGE[1]).all(), msg


class pKdMeasurement(ObservationModelMeasurement):

    r"""
    Measurement where the value(s) come from Kd experiments

    We define the following physics-based function

    $$
    \mathbf{F}_{pK_d}(\Delta g) = - \frac{\Delta g + \ln(C)}{\ln(10)},
    $$
    where C given in molar [M] can be adapted if measurements were undertaken at different concentrations.

    Note
    ----
    The acceptable range for this measurement is [0, 15], inclusive.
    """
    RANGE = (0, 15)

    @staticmethod
    def _observation_model_pytorch(dG_over_KT, standard_conc=1, **kwargs):
        return -(dG_over_KT + np.log(standard_conc)) / LN10

    # implementation does not rely on any torch.* methods so we can just reuse it
    # for other backends via aliases
    _observation_model_numpy = _observation_model_pytorch
    _observation_model_xgboost = _observation_model_pytorch

    @staticmethod
    def _loss_adapter_xgboost_mse(labels, dG_over_KT, standard_conc=1, **kwargs):
        r"""
        Return the gradient and the hessian of the loss defined by

        $$
        L(y, \hat y) = \frac{1}{2} * (y - F(\hat y)) ** 2.
        $$
        """
        grad_loss = (labels + (dG_over_KT + standard_conc) / LN10) / LN10
        hess_loss = np.full(grad_loss.shape, 1 / (LN10 * LN10))

        return grad_loss.astype("float32"), hess_loss.astype("float32")

    @staticmethod
    def _post_loss_adapter(loss, **kwargs):
        #  TODO: Revisit which kind of weighting is needed here
        return loss

    def check(self):
        super().check()
        msg = f"Values for {self.__class__.__name__} are expected to be in the [0, 15] range."
        assert (self.RANGE[0] <= self.values <= self.RANGE[1]).all(), msg


def null_observation_model(arg):
    """
    A callable that returns ``arg`` directly. It works as an
    identity function when observation models need to be disabled
    for a particular experiment.
    """
    import warnings

    warnings.warn(
        "`null_observation_model` is deprecated. "
        "Use `<MeasurementType>.observation_model(backend='null')` instead",
        DeprecationWarning,
    )
    return arg
