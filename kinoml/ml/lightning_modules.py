"""
Training loops built with pytorch-lightning
"""

from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning import metrics
from sklearn.metrics import r2_score

from ..core import measurements as measurement_types
from ..core.measurements import null_observation_model as _null_observation_model
from ..analysis.plots import predicted_vs_observed


class RootMeanSquaredError(metrics.MeanSquaredError):
    def compute(self):
        return torch.sqrt(super().compute())


class ObservationModelModule(pl.LightningModule):
    def __init__(
        self,
        nn_model,
        optimizer,
        loss_function,
        validate=True,
    ):
        super().__init__()
        self.nn_model = nn_model
        # observation model might be reassigned during training
        # to deal with different datasets
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.metric_r2 = r2_score
        self.metric_mae = metrics.MeanAbsoluteError()
        self.metric_mse = metrics.MeanSquaredError()
        # self.metric_rmse = RootMeanSquaredError()

    @property
    def measurement_type_class(self):
        return getattr(measurement_types, self.observation_model.__qualname__.split(".")[0], None)

    def forward(self, x, observation_model=_null_observation_model):
        delta_g = self.nn_model(x)
        prediction = observation_model(delta_g)
        return prediction

    def _standard_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        predicted = self.forward(x, observation_model=batch.observation_model)
        loss = self.loss_function(predicted, y.view_as(predicted))
        return predicted, loss

    def training_step(self, batch, batch_idx, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, **kwargs)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, **kwargs)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0, **kwargs):
        predicted, loss = self._standard_step(batch, batch_idx, **kwargs)
        _, y = batch
        observed = y.view_as(predicted)
        self.log("test_loss", loss)
        self.log("R2", self.metric_r2(predicted, observed))
        self.log("MAE", self.metric_mae(predicted, observed))
        self.log("MSE", self.metric_mse(predicted, observed))
        # self.log("RMSE", self.metric_rmse(predicted, observed))

        # FIXME: See if `batch` can also host the measurement class
        # or change the API to pass the classes around, not the staticmethods
        if self.measurement_type_class is not None:
            plot = predicted_vs_observed(
                predicted, observed, self.measurement_type_class, with_metrics=False
            )
            self.logger.experiment.add_figure("predicted_vs_observed", plot)
        return loss

    def configure_optimizers(self):
        return self.optimizer


class MultiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: List[Dataset],
        observation_models: List[callable] = (_null_observation_model,),
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets

        # If only one observation model is provided, we use the same one for
        # all datasets
        if len(datasets) > 1 and len(observation_models) == 1:
            observation_models = observation_models * len(datasets)

        self.observation_models = observation_models
        self.dataloader_options = kwargs or {}

        self.measurement_types = [om.__qualname__.split(".")[0] for om in observation_models]
        self.dataset_map = dict(zip(self.measurement_types, self.datasets))
        self.observation_model_map = dict(zip(self.measurement_types, self.observation_models))

        self._active_dataset_index = 0

        self.prepare_data()
        self.setup()

    def dataset_indices_by_size(self, reverse=False):
        return sorted(
            range(len(self.datasets)), key=lambda i: len(self.datasets[i]), reverse=reverse
        )

    @property
    def active_dataset(self):
        return self.datasets[self.active_dataset_index]

    @property
    def active_dataset_index(self):
        return self._active_dataset_index

    @active_dataset_index.setter
    def active_dataset_index(self, value):
        assert 0 <= value < len(self.datasets), f"`value` must be in (0, {len(self.datasets)}"
        self._active_dataset_index = value

    def _build_dataloader(self, dataset_index=None, indices=None):
        if dataset_index is None:
            dataset_index = self.active_dataset_index

        if indices is None:
            indices = self.datasets[dataset_index].indices
        dl = ObservationModelDataLoader(
            dataset=self.datasets[dataset_index],
            observation_model=self.observation_models[dataset_index],
            sampler=SubsetRandomSampler(indices),
            **self.dataloader_options,
        )
        return dl

    def train_dataloader(self, dataset_index=None):
        return self._build_dataloader(
            dataset_index=dataset_index, indices=self.datasets[dataset_index].indices["train"]
        )

    def val_dataloader(self, dataset_index=None):
        # return [
        #     self._build_dataloader(dataset_index=i, indices=self.datasets[i].indices["val"])
        #     for i in range(len(self.datasets))
        # ]
        return self._build_dataloader(
            dataset_index=dataset_index, indices=self.datasets[dataset_index].indices["val"]
        )

    def test_dataloader(self, dataset_index=None):
        # return [
        #     self._build_dataloader(dataset_index=i, indices=self.datasets[i].indices["test"])
        #     for i in range(len(self.datasets))
        # ]
        return self._build_dataloader(
            dataset_index=dataset_index, indices=self.datasets[dataset_index].indices["test"]
        )

    def get_kfold(self, *args, **kwargs):
        from sklearn.model_selection import KFold

        # Start with small datasets first to ensure they are seen before overfitting to larger ones
        kfold = KFold(*args, **kwargs)
        for dataset_index in self.dataset_indices_by_size(reverse=True):
            dataset = self.datasets[dataset_index]
            train_val_indices = np.concatenate([dataset.indices["train"], dataset.indices["val"]])

            # Check splitting indices is the same as splitting on the dataset
            for fold_index, (train_index, val_index) in enumerate(kfold.split(train_val_indices)):
                print(
                    f"DS #{dataset_index} {self.measurement_types[dataset_index]}, fold={fold_index}"
                )
                train_dl = self._build_dataloader(dataset_index=dataset_index, indices=train_index)
                val_dl = self._build_dataloader(dataset_index=dataset_index, indices=val_index)
                yield train_dl, val_dl

            # Stop after first one for now
            # break


class CrossValidateTrainer:
    def __init__(self, nfolds=5, *args, **kwargs):
        self.nfolds = nfolds
        self.trainer_args = args
        self.trainer_kwargs = kwargs
        self._models = []
        self._trainers = []

    def fit(self, model, datamodule):
        # .get_kfold() will provide nfolds * len(datamodule.datasets) iterations
        # but we reuse the first nfolds models across the datasets
        for i, (train_loader, val_loader) in enumerate(datamodule.get_kfold(self.nfolds)):
            # This is the first (and maybe only) time we fit something
            # We want to keep them around in case we fit more datamodules (other measurement types)
            fold_index = i % self.nfolds
            if len(self._models) <= self.nfolds:
                fold_model = deepcopy(model)
                self._models.append(fold_model)
                fold_trainer = pl.Trainer(
                    *deepcopy(self.trainer_args), **deepcopy(self.trainer_kwargs)
                )
                self._patch_paths_for_kfold(fold_trainer, i)
                self._trainers.append(fold_trainer)
            else:
                fold_model = self._models[fold_index]
                fold_trainer = self._trainers[fold_index]

            fold_trainer.fit(
                fold_model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader,
            )

    def _patch_paths_for_kfold(self, fold_trainer, fold):
        # Patch filepaths so it contains info about the fold
        # fold_trainer.logger.log_dir += f"/fold_{fold}"
        fold_trainer.logger.experiment.log_dir += f"/fold_{fold}"
        fold_trainer.checkpoint_callback.dirpath += f"/fold_{fold}"
        for path in [
            fold_trainer.logger.log_dir,
            fold_trainer.logger.experiment.log_dir,
            fold_trainer.checkpoint_callback.dirpath,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def test(self, dataset_index=0, **kwargs):
        if "model" in kwargs:
            raise ValueError("`model` option not allowed. Will use internally stored ones.")
        if "test_dataloaders" in kwargs:
            raise ValueError("`test_dataloaders` option not allowed. Provide custom datamodule.")
        datamodule = kwargs.pop("datamodule")
        results = []
        for fold_index, (trainer, model) in enumerate(zip(self._trainers, self._models)):
            model.observation_model = datamodule.observation_models[dataset_index]
            results.append(
                trainer.test(
                    model=model,
                    test_dataloaders=datamodule.test_dataloader(dataset_index=dataset_index),
                    ckpt_path=trainer.checkpoint_callback.best_model_path,
                )
            )

        avg_results = {}
        for key in results[0][0]:
            astensor = np.array([r[key] for rr in results for r in rr])
            avg_results[key] = {"mean": astensor.mean(), "std": astensor.std()}
        return avg_results

    def best_run(self):
        best_index = 0
        best_score = np.inf
        for i, subtrainer in enumerate(self._trainers):
            if subtrainer.checkpoint_callback.best_model_score < best_score:
                best_index, best_score = i, subtrainer.checkpoint_callback.best_model_score
        return self._trainers[best_index]

    def clear(self):
        self._models[:] = []
        self._trainers[:] = []


class ObservationModelDataLoader(DataLoader):
    def __init__(self, observation_model=_null_observation_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_model = observation_model

    def __iter__(self):  # noqa
        iterator = super().__iter__()
        return _IterWithObsModel(iterator, self.observation_model)


class _IterWithObsModel:
    def __init__(self, iterator, observation_model):
        self.iterator = iterator
        self.observation_model = observation_model

    def __getattr__(self, name: str):
        return self.iterator.__getattribute__(name)

    def __next__(self):
        data = self.iterator.__next__()
        return AttrList(data, observation_model=self.observation_model)

    def __iter__(self):
        return self


class AttrList(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.

    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'

    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4

    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]

    Taken from https://code.activestate.com/recipes/579103-python-addset-attributes-to-list/
    """

    def __new__(self, *args, **kwargs):
        return super(AttrList, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
