import torch
from pathlib import Path
import numpy as np
import pandas as pd

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from afa4cate.utils import initialize_wandb
from afa4cate.utils.seed import set_seed_everywhere

from afa4cate.workflows.utils import get_experiment_dir, get_tuning_dir
from afa4cate.workflows.training_cate_models import train_or_load_model, train_or_load_pi_model, train_or_load_variance_model
from afa4cate.workflows.acquisition_metrics import ACQUISITION_METRICS
from afa4cate.experiments.tuning_models import tuning
from afa4cate.experiments.tuning_variance_model import tuning

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


# Main training function
@hydra.main(config_path="../configs", config_name="afa4cate")
def train(config: DictConfig):
    # Temporarily disable strict mode to add new keys
    OmegaConf.set_struct(config.dataset, False)

    # Initialize logging
    if config.wandb_log:
        run_id = initialize_wandb(config)
        logging.info(f"Wandb run ID: {run_id}")

    set_seed_everywhere(config.random_seed)

    if config.dataset.dataset_name == "cmnist":
        root = Path(__file__).parent.parent / "datasets" / "mnist"
        root.mkdir(parents=True, exist_ok=True)
        config.dataset.dataset.root = root
    
    # create the training dataset
    config.dataset.dataset.split = "train"
    ds_train = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "valid"
    ds_valid = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "test"
    ds_test = instantiate(config.dataset).dataset

    if config.tune_cate_model:
        # Get the best hyperparameters
        hyper_dir = Path(__file__).parent / "tuning_files"
        best_param_dir = get_tuning_dir(config.dataset, hyper_dir)
        if not (best_param_dir / "best_hyperparameters.csv").exists():
            tuning(config)
        params = pd.read_csv(best_param_dir / "best_hyperparameters.csv").to_dict(orient="records")[0]
        logging.info("Setting the best hyperparameters for the model.")
        config.cate_model.cate_model.kernel = params["kernel"]
        config.cate_model.cate_model.num_inducing_points = params["num_inducing_points"]
        config.cate_model.cate_model.dim_hidden = params["dim_hidden"]
        config.cate_model.cate_model.depth = params["depth"]
        config.cate_model.cate_model.negative_slope = params["negative_slope"]
        config.cate_model.cate_model.dropout_rate = params["dropout_rate"]
        config.cate_model.cate_model.spectral_norm = params["spectral_norm"]
        config.cate_model.cate_model.learning_rate = params["learning_rate"]
        config.cate_model.cate_model.batch_size = params["batch_size"]

    # Initialize the model
    job_dir = Path(__file__).parent / "saved_files"
    config.job_dir = job_dir
    experiment_dir = get_experiment_dir(config.dataset, job_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving the model and results into the experiment directory: {experiment_dir}")

    model = train_or_load_model(config.cate_model, ds_train, ds_valid, experiment_dir)

    # obtain the estimates of variance
    mu0_train, mu1_train = model.predict_mus(ds_train, posterior_sample=1000)
    mu0_valid, mu1_valid = model.predict_mus(ds_valid, posterior_sample=1000)
    mu0_test, mu1_test = model.predict_mus(ds_test, posterior_sample=1000)

    variance_train = np.var(mu1_train - mu0_train, axis=1)
    variance_valid = np.var(mu1_valid - mu0_valid, axis=1)
    variance_test = np.var(mu1_test - mu0_test, axis=1)

    logging.info("Obtaining the variance estimates for the training, validation, and test sets.")
    ds_train.output_variance(variance_train)
    ds_valid.output_variance(variance_valid)
    ds_test.output_variance(variance_test)

    tuning(config, ds_train, ds_valid)

    # train the variance model
    var_model = train_or_load_variance_model(config.var_model, ds_train, ds_valid, experiment_dir)

    logging.info(f"Running the training with the following acquisition metric: {config.acquisition_metric}")

    print("yaaay!")



if __name__ == "__main__":
    train()