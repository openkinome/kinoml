"""
Training loops built with pytorch-lightning
"""

from itertools import cycle

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ..core.measurements import null_observation_model as _null_observation_model
from ..analysis.plots import predicted_vs_observed


class ObservationModelModule(pl.LightningModule):
    _RESULTS = {"train": pl.TrainResult, "val": pl.EvalResult, "test": pl.EvalResult}

    def __init__(
        self, nn_model, optimizer, loss_function, observation_model=_null_observation_model
    ):
        super().__init__()
        self.nn_model = nn_model
        self.observation_model = observation_model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def forward(self, x):
        delta_g = self.nn_model(x)
        prediction = self.observation_model(delta_g)
        return prediction

    def _standard_step(self, batch, batch_idx, kind="train", **kwargs):
        assert kind in self._RESULTS
        x, y = batch
        predicted = self.forward(x)
        loss = self.loss_function(predicted, y.view(*predicted.shape))
        return predicted, loss

    def training_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, kind="train", **kwargs)
        result = pl.TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
        result.log("train_loss", loss, on_step=True, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, kind="val", **kwargs)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result

    def test_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, kind="test", **kwargs)
        observed = batch[1].view(*predicted.shape)
        result = pl.EvalResult()
        result.test_loss = loss
        result.log("R2", pl.metrics.sklearns.R2Score()(predicted, observed))
        result.log("MAE", pl.metrics.functional.mae(predicted, observed))
        result.log("MSE", pl.metrics.functional.mse(predicted, observed))
        result.log("RMSE", pl.metrics.functional.rmse(predicted, observed))
        result.plot = predicted_vs_observed(predicted, observed, with_metrics=False)
        self.logger.experiment.add_figure("predicted_vs_observed", result.plot)
        return result

    def configure_optimizers(self):
        return self.optimizer

