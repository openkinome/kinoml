"""
Common plots for ML model performance analysis
"""

import numpy as np
from matplotlib import pyplot as plt
from .metrics import performance


def predicted_vs_observed(predicted, observed, measurement_type, with_metrics=True, **kwargs):
    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(predicted, observed)

    limits = np.array(measurement_type.RANGE)
    padded_limits = limits[0] - 0.05 * limits.max(), limits[1] + 0.05 * limits.max()

    ax.set(xlim=padded_limits, ylim=padded_limits)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    x = np.linspace(padded_limits[0], padded_limits[1], 100)
    ax.plot(x, x)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{predicted.shape[0]} {measurement_type.__name__}")

    if with_metrics:
        performance(predicted, observed, **kwargs)
    plt.close()
    return fig
