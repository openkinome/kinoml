"""
WIP
"""
from tqdm.auto import trange
import torch


def multi_measurement_training_loop(
    dataloaders, observation_models, model, optimizer, loss_function, epochs=100
):
    """
    Standard training loop with multiple dataloaders and observation models.

    Parameters
    ----------
    dataloaders: dict of str -> torch.utils.data.DataLoader
        key must refer to the measurement type present in the dataloader
    observation_models: dict of str -> callable
        keys must be the same as in dataloaders, and point to pytorch-compatible callables
        that convert delta_over_kt to the corresponding measurement_type
    model: torch.nn.Model
        instance of the model to train
    optimizer: torch.optim.Optimizer
        instance of the optimization engine
    loss_function: torch.nn.modules._Loss
        instance of the loss function to apply (e.g. MSELoss())
    epochs: int
        number of iterations the loop will run

    Returns
    -------
    model: torch.nn.Model
        The trained model (same instance as provided in parameters)
    loss_timeseries: list of float
        Cumulative loss per epoch
    """
    msg = "Keys in `dataloaders` must be same or a subset of those in `observation_models`"
    assert set(dataloaders.keys()).issubset(set(observation_models.keys())), msg

    loss_timeseries = []
    range_epochs = trange(epochs, desc="Epochs")
    for epoch in range_epochs:
        # TODO: Single cumulative loss / or loss per loader? look into this!
        cumulative_loss = 0.0
        for key, loader in dataloaders.items():
            for x, y in loader:
                # Clear gradients
                optimizer.zero_grad()

                # Obtain model prediction given model input
                delta_g = model(x)

                # apply observation model
                prediction = []
                for dG, func in zip(delta_g, post_prediction_callables):
                    prediction.append(func(dG))
                prediction = torch.tensor(prediction)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # !!! Make sure prediction and y match shapes !!!
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                y = y.reshape(prediction.shape)

                loss = loss_function(prediction, y)

                # Obtain loss for the predicted output
                cumulative_loss += loss.item()

                # Gradients w.r.t. parameters
                loss.backward()

                # Optimize
                optimizer.step()

        range_epochs.set_description(f"Epochs (loss={cumulative_loss:.2e})")
        loss_timeseries.append(cumulative_loss)

    return model, loss_timeseries


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Taken from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.

    Taken from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


def _old_training_loop():
    """Deprecated"""
    # from kinoml.ml.lightning_modules import KFold3Way, KFold
    # from IPython.display import Markdown
    # from tqdm.auto import trange, tqdm
    # from kinoml.ml.torch_models import NeuralNetworkRegression
    # from ipywidgets import HBox, VBox, Output, HTML
    # from kinoml.analysis.plots import predicted_vs_observed, performance
    # from kinoml.utils import fill_until_next_multiple
    # import pandas as pd
    # import torch.nn as nn

    # if VALIDATION:
    #     kfold = KFold3Way(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS)
    #     ttypes = ["train", "val", "test"]
    # else:
    #     kfold = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLDS)
    #     ttypes = ["train", "test"]

    # ModelCls = import_object(MODEL_CLS)

    # kinase_metrics = defaultdict(dict)
    # for dataset in tqdm(DATASETS):
    #     name = dataset.metadata["measurement_type"]
    #     mtype = import_object(f"kinoml.core.measurements.{name}")
    #     if dataset.shape_X[0] < MIN_ITEMS_PER_DATASET:
    #         warn(f"Ignoring {name} because it has less than {MIN_ITEMS_PER_DATASET}")
    #         continue

    #     if VERBOSE:
    #         display(Markdown(f"#### {name}"))

    #     metrics = defaultdict(list)

    #     ds_size = list(range(dataset.shape_y[0]))
    #     for fold_index, splits in enumerate(kfold.split(ds_size, ds_size)):
    #         if VALIDATION:
    #             train_indices, val_indices, test_indices = splits
    #         else:
    #             train_indices, test_indices = splits

    #         if VERBOSE:
    #             display(Markdown(f"##### Fold {fold_index}"))

    #         ####
    #         # TRAIN
    #         ####
    #         x_train, y_train = dataset[train_indices]
    #         x_test, y_test = dataset[test_indices]
    #         if VALIDATION:
    #             x_val, y_val = dataset[val_indices]

    #         if ModelCls.needs_input_shape:
    #             MODEL_KWARGS["input_shape"] = ModelCls.estimate_input_shape(x_train)
    #         nn_model = ModelCls(**MODEL_KWARGS)
    #         nn_model.train(True)

    #         optimizer = torch.optim.Adam(
    #             nn_model.parameters(), lr=LEARNING_RATE, eps=EPSILON, betas=BETAS
    #         )
    #         loss_function = torch.nn.MSELoss()

    #         if VERBOSE:
    #             range_epochs = trange(MAX_EPOCHS, desc="Epochs (+ featurization...)")
    #         else:
    #             range_epochs = range(MAX_EPOCHS)
    #         for epoch in range_epochs:
    #             optimizer.zero_grad()

    #             prediction = nn_model(x_train)
    #             if WITH_OBSERVATION_MODEL:
    #                 prediction = mtype.observation_model(backend="pytorch")(prediction)
    #             prediction = prediction.view_as(y_train)

    #             loss = loss_function(prediction, y_train)
    #             if VERBOSE:
    #                 range_epochs.set_description(f"Epochs (loss={loss.item():.2e})")

    #             if VALIDATION:
    #                 warn("Validation step not implemented yet")

    #             # Gradients w.r.t. parameters
    #             loss.backward()

    #             # Optimizer
    #             optimizer.step()

    #         ###
    #         # Save model's state -- you will still need to instantiate the model class!
    #         # Possibly using something like:
    #         # model = import_object(MODEL_CLS)(**MODEL_KWARGS)
    #         # model.load_state_dict(torch.load("state_dict.pt"))
    #         ###
    #         torch.save(nn_model.state_dict(), OUT / f"state_dict_{name}_fold{fold_index}.pt")

    #         ####
    #         # EVAL
    #         ####
    #         nn_model.eval()
    #         outputs = []
    #         for ttype in ttypes:
    #             output = Output()
    #             with output:
    #                 title = f"fold={fold_index}, {ttype}={locals()[f'{ttype}_indices'].shape[0]}"
    #                 print(title)
    #                 print("-" * (len(title)))

    #                 observed = locals()[f"y_{ttype}"]

    #                 with torch.no_grad():
    #                     predicted = nn_model(locals()[f"x_{ttype}"])
    #                     if WITH_OBSERVATION_MODEL:
    #                         predicted = mtype.observation_model(backend="pytorch")(predicted)

    #                 predicted = predicted.view_as(observed).detach().numpy()
    #                 observed = observed.detach().numpy()
    #                 these_metrics = performance(
    #                     predicted, observed, n_boot=N_BOOTSTRAPS, sample_ratio=BOOTSTRAP_SAMPLE_RATIO
    #                 )
    #                 metrics[ttype].append(these_metrics)
    #                 # if VERBOSE:
    #                 #     display(predicted_vs_observed(predicted, observed, mtype_class, with_metrics=False))

    #             outputs.append(output)
    #         if VERBOSE:
    #             display(HBox(outputs))

    #     # Average performances

    #     average = defaultdict(dict)
    #     for key in metrics["test"][0]:
    #         for label in ttypes:
    #             # this zero here ---v is super important! we only want the mean of the means!
    #             values = [fold[key][0] for fold in metrics[label]]
    #             average[label][key] = {"mean": np.mean(values), "std": np.std(values)}
    #     if VERBOSE:
    #         for label in ttypes:
    #             display(HTML(f"Bootstrapped average across folds ({label}):"))
    #             display(pd.DataFrame.from_dict(average[label]))
    #     kinase_metrics[name] = average
