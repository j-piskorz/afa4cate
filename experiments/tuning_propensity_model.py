import torch
from pathlib import Path
import numpy as np
import pandas as pd

from ray import tune
from ray.tune import schedulers
from ray.tune.search import hyperopt

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from afa4cate.utils import initialize_wandb

from afa4cate.cate_models import DeepKernelGP
from afa4cate.datasets import SyntheticCATEDataset
from afa4cate.workflows.utils import get_tuning_dir
from afa4cate.workflows.training_cate_models import train_or_load_pi_model

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


# Main training function
@hydra.main(config_path="../configs", config_name="afa4cate")
def tuning(config: DictConfig):
    # Temporarily disable strict mode to add new keys
    OmegaConf.set_struct(config.dataset, False)
    
    # Initialize the model
    job_dir = Path(__file__).parent / "tuning_files_pi"
    config.job_dir = job_dir
    experiment_dir = get_tuning_dir(config.dataset, job_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving the model and results into the experiment directory: {experiment_dir}")

    space = {
        "dim_hidden": tune.choice([128, 256, 512]),
        "depth": tune.choice([2, 3, 4]),
        "negative_slope": tune.choice([-100.0, -1.0, 0.0, 0.2]),
        "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.5]),
        "spectral_norm": tune.choice([0.0, 0.95, 1.5, 3.0]),
        "learning_rate": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([128, 256]),
    }

    # create the training dataset
    config.dataset.dataset.mode="pi"
    config.dataset.dataset.split = "train"
    ds_train_pi = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "valid"
    ds_valid_pi = instantiate(config.dataset).dataset

    def func(params):
        config.pi_model.dim_hidden = params["dim_hidden"]
        config.pi_model.depth = params["depth"]
        config.pi_model.negative_slope = params["negative_slope"]
        config.pi_model.dropout_rate = params["dropout_rate"]
        config.pi_model.spectral_norm = params["spectral_norm"]
        config.pi_model.learning_rate = params["learning_rate"]
        config.pi_model.batch_size = params["batch_size"]

        _ = train_or_load_pi_model(config.pi_model, ds_train_pi, ds_valid_pi, None)

    algorithm = hyperopt.HyperOptSearch(
        space, metric="mean_loss", mode="min", n_initial_points=100,
    )
    scheduler = schedulers.AsyncHyperBandScheduler(
        grace_period=100, max_t=config.tune.epochs
    )
    analysis = tune.run(
        run_or_experiment=func,
        metric="mean_loss",
        mode="min",
        name="hyperopt_deep_kernel_gp",
        resources_per_trial={
            "cpu": config.tune.cpu_per_trial,
            "gpu": config.tune.gpu_per_trial,
        },
        num_samples=config.tune.max_samples,
        time_budget_s=config.tune.time_budget_s,
        search_alg=algorithm,
        scheduler=scheduler,
        storage_path=experiment_dir,
        raise_on_failed_trial=False,
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    results = pd.DataFrame(analysis.best_config, index=[0])
    results.to_csv(experiment_dir / "best_hyperparameters_pi.csv")


if __name__ == "__main__":
    tuning()