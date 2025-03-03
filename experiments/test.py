import torch
from pathlib import Path
import numpy as np
import pandas as pd
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from afa4cate.utils import initialize_wandb
from afa4cate.utils.seed import set_seed_everywhere

from afa4cate.workflows.utils import get_experiment_dir, get_tuning_dir
from afa4cate.workflows.training_cate_models import train_or_load_model, train_or_load_pi_model
from afa4cate.workflows.acquisition_metrics import ACQUISITION_METRICS
from afa4cate.experiments.tuning_models import tuning

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
    elif config.dataset.dataset_name == "ihdp" or config.dataset.dataset_name == 'ihdp_cov':
        root = Path(__file__).parent.parent / "datasets" / "ihdp"
        root.mkdir(parents=True, exist_ok=True)
        config.dataset.dataset.root = root

    # create the training dataset
    config.dataset.dataset.split = "train"
    ds_train = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "valid"
    ds_valid = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "test"
    ds_test = instantiate(config.dataset).dataset

    logging.info(f"The proportion of samples who were assigned the treatment: {ds_train.t.mean()}")

    if config.tune_cate_model:
        # Get the best hyperparameters
        hyper_dir = Path(__file__).parent / "tuning_files" / config.cate_model.model_name
        best_param_dir = get_tuning_dir(config.dataset, hyper_dir)
        if not (best_param_dir / "best_hyperparameters.csv").exists():
            tuning(config)
        params = pd.read_csv(best_param_dir / "best_hyperparameters.csv").to_dict(orient="records")[0]
        logging.info("Setting the best hyperparameters for the model.")
        if config.cate_model.model_name == "deep_kernel_gp":
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
    mu0_pred_full, mu1_pred_full = model.predict_mus(ds_test)
    tau_pred_full = mu1_pred_full - mu0_pred_full

    if config.wandb_log:
        wandb.log({"pehe_full": np.sqrt(np.mean(tau_pred_full - ds_test.tau)**2)})

    pehe, pehe_percentage, mu_0_mse, mu_1_mse, y_mse = model.compute_pehe(ds_train, seed=config.random_seed)

    logging.info(f"PEHE: {pehe}")
    logging.info(f"PEHE Percentage: {pehe_percentage}")
    logging.info(f"Mu_0 RMSE: {mu_0_mse}")
    logging.info(f"Mu_1 RMSE: {mu_1_mse}")
    logging.info(f"Y_MSE: {y_mse}")

    logging.info(f"Loss train: {model.calculate_loss(ds_train)}")
    logging.info(f"Loss valid: {model.calculate_loss(ds_valid)}")

    if (config.acquisition_metric == "r_WPO_total"
        or config.acquisition_metric == "r_WPO_mean"
        or config.acquisition_metric == "r_WPO"
        or config.acquisition_metric == "r_WPO_var"):
        # get the propensity datasets
        config.dataset.dataset.mode = "pi"
        config.dataset.dataset.split = "train"
        ds_pi_train = instantiate(config.dataset).dataset
        config.dataset.dataset.split = "valid"
        ds_pi_valid = instantiate(config.dataset).dataset
        config.dataset.dataset.split = "test"
        ds_pi_test = instantiate(config.dataset).dataset

        # get the propensity model
        pi_model = train_or_load_pi_model(config.pi_model, ds_pi_train, ds_pi_valid, experiment_dir)

        # check the propensity model error
        pi_pred_full = pi_model.predict_mean(ds_pi_test)
        if config.dataset.dataset_name == "synthetic":
            pi_rmse = np.sqrt(np.mean((pi_pred_full - ds_pi_test.pi)**2))
            logging.info(f"Propensity model RMSE: {pi_rmse}")
            if config.wandb_log:
                wandb.log({"pi_rmse": pi_rmse})
    else:
        pi_model = None

    x_test = ds_test.x
    tau_test = ds_test.tau
    n, d = x_test.shape


if __name__ == "__main__":
    train()