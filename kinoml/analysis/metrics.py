import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def root_mean_squared_error(*args, **kwargs):
    return np.sqrt(mean_squared_error(*args, **kwargs))


def performance(
    predicted, observed, verbose=True, n_boot=100, confidence=0.95, sample_ratio=0.8, _seed=1234
):
    """
    TODO: Reimplement samples with scipy.stats.norm or with numpy.

    """
    assert n_boot >= 2, "Number of bootstraps must be >= 2"
    assert 0.5 <= confidence < 1, "Confidence must be in [0.5, 1)"
    assert 0 < sample_ratio < 1, "Sample ration must be in (0, 1)"

    high = predicted.shape[0]
    size = int(sample_ratio * high)
    bootstrapped = np.empty((4, n_boot))
    metrics = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
    }

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
                f"{key.upper():>4s}: {mean:.2f}±{std:.2f} {100*confidence:.0f}CI=({low:.2f}, {high:.2f})"
            )

    return results
