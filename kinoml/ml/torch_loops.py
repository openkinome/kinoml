from tqdm.auto import trange


def multi_measurement_training_loop(
    dataloaders, observation_models, model, optimizer, loss_function, epochs=100
):
    """
    Standard training loop with multiple dataloaders and observation models.

    Parameters:
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

    Returns:
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
        for measurement_type, loader in dataloaders.items():
            obs_model = observation_models[measurement_type]
            for x, y in loader:
                # Clear gradients
                optimizer.zero_grad()

                # Obtain model prediction given model input
                delta_g = model(x)

                # apply observation model
                prediction = obs_model(delta_g)

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
