"""
Training loops built with pytorch-lightning
"""

from itertools import cycle

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import metrics
from sklearn.metrics import r2_score

from ..core.measurements import null_observation_model as _null_observation_model
from ..analysis.plots import predicted_vs_observed


class RootMeanSquaredError(metrics.MeanSquaredLogError):
    def compute(self):
        return torch.sqrt(super().compute())


class ObservationModelModule(pl.LightningModule):
    def __init__(
        self,
        nn_model,
        optimizer,
        loss_function,
        observation_model=_null_observation_model,
        validate=True,
    ):
        super().__init__()
        self.nn_model = nn_model
        # observation model might be reassigned during training
        # to deal with different datasets
        self.observation_model = observation_model
        self.measurement_type_class = None
        self.optimizer = optimizer
        self.loss_function = loss_function
        if validate:
            self.validation_step = self._disabled_validation_step

        self.metric_r2 = r2_score
        self.metric_mae = metrics.MeanAbsoluteError()
        self.metric_mse = metrics.MeanSquaredError()
        self.metric_rmse = RootMeanSquaredError()

    def forward(self, x):
        delta_g = self.nn_model(x)
        prediction = self.observation_model(delta_g)
        return prediction

    def _standard_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        predicted = self.forward(x)
        loss = self.loss_function(predicted, y.view_as(predicted))
        return predicted, loss

    def training_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, **kwargs)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def _disabled_validation_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, **kwargs)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, **kwargs)
        observed = batch[1].view(*predicted.shape)
        self.log("test_loss", loss)
        self.log("R2", self.metric_r2(predicted, observed))
        self.log("MAE", self.metric_mae(predicted, observed))
        self.log("MSE", self.metric_mse(predicted, observed))
        self.log("RMSE", self.metric_rmse(predicted, observed))

        if self.measurement_type_class is not None:
            plot = predicted_vs_observed(
                predicted, observed, self.measurement_type_class, with_metrics=False
            )
            self.logger.experiment.add_figure("predicted_vs_observed", plot)
        return loss

    def configure_optimizers(self):
        return self.optimizer
