import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _rmse(*args, **kwargs):
    return np.sqrt(mean_squared_error(*args, **kwargs))


def performance(predicted, observed, verbose=True, n_boot=100, confidence=0.95, sample_ratio=0.8):
    assert n_boot >= 100, "Number of bootstraps must be >= 100"
    assert 0 < confidence < 1, "Confidence must be in (0, 1)"
    assert 0 < sample_ratio < 1, "Sample ration must be in (0, 1)"

    high = predicted.shape[0]
    size = int(sample_ratio * high)
    bootstrapped = np.empty((4, n_boot))
    metrics = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "rmse": _rmse,
    }
    for i in range(n_boot):
        indices = np.random.randint(low=0, high=high, size=size)
        obs, pred = observed[indices], predicted[indices]
        for j, (key, fn) in enumerate(sorted(metrics.items())):
            bootstrapped[j][i] = fn(obs, pred)

    # FIXME: Sort metrics as suggested here https://stackoverflow.com/a/40491405
    bootstrapped.sort(axis=1)

    high_ci = int(confidence * n_boot)
    low_ci = int(n_boot - high_ci)
    results = {}
    for index, key in enumerate(sorted(metrics)):
        arr = bootstrapped[index]
        results[key] = mean, std, low, high = arr.mean(), arr.std(), arr[low_ci], arr[high_ci]
        if verbose:
            print(
                f"{key.upper():>4s}: {mean:.2f}±{std:.2f} {100*confidence:.0f}CI=({low:.2f}, {high:.2f})"
            )

    return results
