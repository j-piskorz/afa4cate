import numpy as np
import pandas as pd

import optuna

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from afa4cate.utils import initialize_wandb

from afa4cate.workflows.utils import get_tuning_dir
from afa4cate.workflows.training_cate_models import train_or_load_model

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


def tuning(config: DictConfig, hyper_dir):
    # Temporarily disable strict mode to add new keys
    OmegaConf.set_struct(config.dataset, False)
    
    # Initialize the model
    job_dir = hyper_dir
    config.job_dir = job_dir
    experiment_dir = get_tuning_dir(config.dataset, job_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving the model and results into the experiment directory: {experiment_dir}")

    # Create training & validation datasets
    config.dataset.dataset.split = "train"
    ds_train = instantiate(config.dataset).dataset

    config.dataset.dataset.split = "valid"
    ds_valid = instantiate(config.dataset).dataset

    def objective(trial):
        """Objective function for Optuna hyperparameter tuning"""
        params = {
            "dim_hidden": trial.suggest_categorical("dim_hidden", [128, 256, 512]),
            "depth": trial.suggest_categorical("depth", [2, 3, 4]),
            "negative_slope": trial.suggest_categorical("negative_slope", [-1.0, 0.0, 0.1, 0.2]),
            "dropout_rate": trial.suggest_categorical("dropout_rate", [0.05, 0.1, 0.2, 0.5]),
            "spectral_norm": trial.suggest_categorical("spectral_norm", [0.0, 0.95, 1.5, 3.0]),
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "kernel": trial.suggest_categorical("kernel", ["RBF", "Matern12", "Matern32", "Matern52"]),
            "num_inducing_points": trial.suggest_categorical("num_inducing_points", [100, 200, 500])
        }

        # Update config with sampled parameters
        config.cate_model.cate_model.update(params)
        config.cate_model.cate_model.seed = 0  # Fix seed for tuning

        # Train and evaluate the model
        try:
            model = train_or_load_model(config.cate_model, ds_train, ds_valid, None, device="cuda:0", tune=True)
            loss = model.calculate_loss(ds_valid)
        except Exception as e:
            logging.error(f"Error during training: {e}")
            loss = np.inf

        return loss  # Optuna minimizes this metric

    # Run the hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.tune.max_samples, timeout=config.tune.time_budget_s, n_jobs=-1)

    # Save best parameters
    best_params = study.best_params
    print("Best hyperparameters found:", best_params)

    results_df = pd.DataFrame([best_params])
    results_df.to_csv(experiment_dir / "best_hyperparameters.csv", index=False)