"""
Common plots for ML model performance analysis
"""

import numpy as np
from matplotlib import pyplot as plt
from .metrics import performance


def predicted_vs_observed(
    predicted, observed, limits=(0, 15), title=None, with_metrics=True, **kwargs
):
    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(predicted, observed)
    ax.set(xlim=limits, ylim=limits)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    x = np.linspace(limits[0], limits[1], 100)
    ax.plot(x, x)
    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title)
    if with_metrics:
        performance(predicted, observed, **kwargs)
    plt.close()
    return fig
