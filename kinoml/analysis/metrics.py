import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def root_mean_squared_error(*args, **kwargs):
    """
    Returns the square-root of ``scikit-learn``'s ``mean_squared_error`` metric.
    All arguments are forwarded to that function.
    """
    return np.sqrt(mean_squared_error(*args, **kwargs))


def performance(
    predicted, observed, verbose=True, n_boot=100, confidence=0.95, sample_ratio=0.8, _seed=1234
):
    """
    Measure the predicted vs observed performance with different metrics (R2, MSE, MAE, RMSE).

    Parameters
    ----------
    predicted : array-like
        Data points predicted by the model.
    observed : array-like
        Observed data points, as available in the dataset.
    verbose : bool, optional=True
        Whether to print results to stdout.
    n_boot : int, optional=100
        Number of bootstrap iterations. Set to ``1`` to disable
        bootstrapping.
    confidence : float, optional=0.95
        Confidence interval, relative to 1. Default is 95%.
    sample_ratio : float, optional=0.8
        Proportion of data to sample in each iteration.
    _seed : int, optional=1234
        Random seed. Each bootstrap iteration gets a different seed
        based on this initial one.

    Returns
    -------
    results : dict of tuple
        This dictionary contains one item per metric (see above),
        with a 4-element tuple each: mean, standard deviation, and lower and
        upper bounds for the confidence interval.

    Note
    ----
    **TODO**: Reimplement samples with ``scipy.stats.norm`` or with ``numpy``.

    """
    assert 0.5 <= confidence < 1, "Confidence must be in [0.5, 1)"
    assert 0 < sample_ratio <= 1, "Sample ratio must be in (0, 1]"

    high = predicted.shape[0]
    size = int(sample_ratio * high)
    metrics = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
    }
    bootstrapped = np.empty((len(metrics), n_boot))

    for i in range(n_boot):
        rng = np.random.RandomState(_seed + i)
        indices = rng.randint(low=0, high=high, size=size)
        obs, pred = observed[indices], predicted[indices]
        for j, (key, fn) in enumerate(sorted(metrics.items())):
            bootstrapped[j][i] = fn(obs, pred)

    # FIXME: Sort metrics as suggested here https://stackoverflow.com/a/40491405
    bootstrapped.sort(axis=1)

    results = {}
    for index, key in enumerate(sorted(metrics)):
        arr = bootstrapped[index]

        results[key] = mean, std, low, high = (
            arr.mean(),
            arr.std(),
            np.quantile(arr, 1 - confidence),
            np.quantile(arr, confidence),
        )
        if verbose:
            print(
                f"{key.upper():>4s}: {mean:.4f}±{std:.4f} {100*confidence:.0f}CI=({low:.4f}, {high:.4f})"
            )

    return results
