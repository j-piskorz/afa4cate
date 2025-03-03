import numpy as np
import logging
from pathlib import Path

from afa4cate.cate_models import DeepKernelGP
from afa4cate.cate_models.neural_network import NeuralNetwork

def train_or_load_model(config, ds_train, ds_valid, experiment_dir, device=None, tune=False):
    if device is None:
        device = config.cate_model.device
    if experiment_dir is not None:
        out_dir = experiment_dir / "checkpoints"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = get_directory(out_dir, config)
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_dir = None

    if config.model_name == "deep_kernel_gp":
        model = DeepKernelGP(
                job_dir=model_dir,
                kernel=config.cate_model.kernel,
                num_inducing_points=config.cate_model.num_inducing_points,
                inducing_point_dataset=ds_train,
                architecture="resnet",
                dim_input=ds_train.dim_input,
                dim_hidden=config.cate_model.dim_hidden,
                dim_output=config.cate_model.dim_output,
                depth=config.cate_model.depth,
                negative_slope=config.cate_model.negative_slope,
                batch_norm=config.cate_model.batch_norm,
                spectral_norm=config.cate_model.spectral_norm,
                dropout_rate=config.cate_model.dropout_rate,
                weight_decay=(0.5 * (1 - config.cate_model.dropout_rate)) / len(ds_train),
                learning_rate=config.cate_model.learning_rate,
                batch_size=config.cate_model.batch_size,
                epochs= config.cate_model.epochs if not tune else 75,
                patience=config.cate_model.patience,
                num_workers=config.cate_model.num_workers,
                seed=config.cate_model.seed,
                device=device,
            )
        if (experiment_dir is None) or tune:
            _ = model.fit(ds_train, ds_valid, tune=tune)
        elif not (model_dir / "best_checkpoint.pt").exists():
            logging.info(f"Model does not yet exist. Training the model from the data.")
            _ = model.fit(ds_train, ds_valid, tune=tune)
        else:
            logging.info(f"Model already exists. Loading the model parameters.")
            model.load(tune=tune)
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")
    
    return model


def train_or_load_pi_model(config, ds_pi_train, ds_pi_valid, experiment_dir, device=None):
    if device is None:
        device = config.device
    if experiment_dir is not None:
        pi_dir = experiment_dir / "pi"
    else:
        pi_dir = None

    pi_model = NeuralNetwork(
        job_dir=pi_dir,
        architecture="resnet",
        dim_input=ds_pi_train.dim_input,
        dim_hidden=config.dim_hidden,
        dim_output=1,
        depth=config.depth,
        negative_slope=config.negative_slope,
        batch_norm=False,
        spectral_norm=config.spectral_norm,
        dropout_rate=config.dropout_rate,
        weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_pi_train),
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        num_workers=0,
        seed=config.seed,
        device=device,
    )
    if experiment_dir is None:
        _ = pi_model.fit(ds_pi_train, ds_pi_valid)
    elif not (pi_dir / "best_checkpoint.pt").exists():
        pi_model.fit(ds_pi_train, ds_pi_valid)
    else:
        pi_model.load()

    return pi_model


def get_directory(base_dir, config):
    if config.model_name == "deep_kernel_gp":
        # Get model parameters from config
        kernel = config.cate_model.kernel
        num_inducing_points = config.cate_model.num_inducing_points
        dim_hidden = config.cate_model.dim_hidden
        dim_output = config.cate_model.dim_output
        depth = config.cate_model.depth
        negative_slope = config.cate_model.negative_slope
        dropout_rate = config.cate_model.dropout_rate
        spectral_norm = config.cate_model.spectral_norm
        learning_rate = config.cate_model.learning_rate
        batch_size = config.cate_model.batch_size
        epochs = config.cate_model.epochs
        return (
            Path(base_dir)
            / "deep_kernel_gp"
            / f"kernel-{kernel}_ip-{num_inducing_points}-dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
        )
    else:
        raise ValueError(f"Model name {config.cate_model.model_name} not recognized.")
