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
from afa4cate.workflows.tuning_cate_models import tuning

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
        
    # create the training dataset
    config.dataset.dataset.split = "train"
    ds_train = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "valid"
    ds_valid = instantiate(config.dataset).dataset
    config.dataset.dataset.split = "test"
    ds_test = instantiate(config.dataset).dataset

    logging.info(f"The proportion of samples who were assigned the treatment: {ds_train.t.mean()}")

    # Get the best hyperparameters
    hyper_dir = Path(__file__).parent / "tuning_files"
    best_param_dir = get_tuning_dir(config.dataset, hyper_dir)
    if not (best_param_dir / "best_hyperparameters.csv").exists():
        tuning(config, hyper_dir)
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
    mu0_pred_full, mu1_pred_full = model.predict_mus(ds_test, posterior_sample=1000)
    mu0_mean_full = mu0_pred_full.mean(axis=1)
    mu1_mean_full = mu1_pred_full.mean(axis=1)
    tau_pred_full = mu1_mean_full - mu0_mean_full

    # calculating the variance threshold using the train set
    mu0_train, mu1_train = model.predict_mus(ds_train, posterior_sample=1000)
    if config.acquisition.threshold_variable == "TE":
        variance_dist = (mu1_train - mu0_train).var(axis=1)
        var_threshold = np.quantile(variance_dist, config.acquisition.percentile_var_threshold)
        logging.info(f"Variance threshold (based on TE): {var_threshold}")
    elif config.acquisition.threshold_variable == "PO":
        variance_dist_1 = mu1_train.var(axis=1)
        variance_dist_0 = mu0_train.var(axis=1)
        var_threshold_1 = np.quantile(variance_dist_1, config.acquisition.percentile_var_threshold)
        var_threshold_0 = np.quantile(variance_dist_0, config.acquisition.percentile_var_threshold)
        var_threshold = np.min([var_threshold_1, var_threshold_0])
        logging.info(f"Variance threshold (based on PO): {var_threshold}")
    elif config.acquisition.threshold_variable == "sTE":
        variance_dist = (mu1_train - mu0_train > 0).var(axis=1)
        var_threshold = np.quantile(variance_dist, config.acquisition.percentile_var_threshold)
        logging.info(f"Variance threshold (based on sTE): {var_threshold}")
    else:
        raise ValueError("Invalid threshold variable")

    if config.wandb_log:
        wandb.log({"pehe_full": np.sqrt(np.mean(tau_pred_full - ds_test.tau)**2)})
        wandb.log({"var_threshold": var_threshold})

    pehe, pehe_percentage, mu_0_mse, mu_1_mse, y_mse = model.compute_pehe(ds_train, seed=config.random_seed)

    logging.info(f"PEHE: {pehe}")
    logging.info(f"PEHE Percentage: {pehe_percentage}")
    logging.info(f"Mu_0 RMSE: {mu_0_mse}")
    logging.info(f"Mu_1 RMSE: {mu_1_mse}")
    logging.info(f"Y_MSE: {y_mse}")

    logging.info(f"Loss train: {model.calculate_loss(ds_train)}")
    logging.info(f"Loss valid: {model.calculate_loss(ds_valid)}")
    
    x_test = ds_test.x
    tau_test = ds_test.tau
    n, d = x_test.shape

    # select the initially observed variables
    num_ones = 3 if config.dataset.dataset_name == "acic2016" else 7
    z_test = np.zeros((n, d), dtype=int)  # Initialize a matrix of zeros
    # Generate the indices where the 1s will go
    rows = np.repeat(np.arange(n), num_ones)  # Repeat row indices num_ones times
    cols = np.array([np.random.choice(d, num_ones, replace=False) for _ in range(n)]).flatten()  # Randomly choose column indices
    # Set the corresponding locations to 1
    z_test[rows, cols] = 1
    
    gsm = instantiate(config.gs_model).gs_model
    if config.dataset.dataset_name == "ihdp":
        gsm.define_mean_sigma(mu=ds_train.mean, Sigma=ds_train.cov)
    acquisition_metric = ACQUISITION_METRICS[config.acquisition_metric]

    # select the subset of the test set to run the acquisition
    if config.acquisition.subsample == 'random':
        test_subset = np.random.choice(ds_test.x.shape[0], config.acquisition.n_test_samples, replace=False)
    elif config.acquisition.subsample == 'good_propensity':
        if config.dataset.dataset_name != "acic2016":
            raise ValueError("This subsample method is only available for the acic2016 dataset.")
        propensities = np.abs(ds_test.pi - 0.5)
        test_subset = np.argsort(propensities)[:config.acquisition.n_test_samples]
    elif config.acquisition.subsample == 'bad_propensity':
        if config.dataset.dataset_name != "acic2016":
            raise ValueError("This subsample method is only available for the acic2016 dataset.")
        propensities = np.abs(ds_test.pi - 0.5)
        test_subset = np.argsort(propensities)[-config.acquisition.n_test_samples:]
    elif config.acquisition.subsample == 'bad_variance':
        var_dist_test = np.var(mu1_pred_full - mu0_pred_full, axis=1)
        test_subset = np.argsort(var_dist_test)[-config.acquisition.n_test_samples:]
    elif config.acquisition.subsample == 'bad_initial_estimate':
        # first select a subsample of the test set, to avoid using the entire test set
        init_test_subset = np.random.choice(ds_test.x.shape[0], (config.acquisition.n_test_samples*5), replace=False)
        init_pehes = []
        for i in init_test_subset:
            # evaluating the model after the acquisition
            logging.info(f"Running initial estimate for individual {i}.")
            X_torch = torch.from_numpy(x_test[i]).float().unsqueeze(0)
            Z_torch = torch.from_numpy(z_test[i]).float()
            new_x_samples = gsm.sample_conditional_covariates(X_torch, Z_torch, n_samples=config.acquisition.n_cond_samples, seed=config.random_seed)
            mu0_samples, mu1_samples = model.predict_mus_from_covariates(new_x_samples, posterior_sample=config.acquisition.n_posterior_samples,
                                                                         batch_size=16384)
            mu0_samples = mu0_samples[0]
            mu1_samples = mu1_samples[0]
            tau_pred = (mu1_samples - mu0_samples).mean()
            pehe = (tau_pred - tau_test[i])**2
            init_pehes.append(pehe)
        test_subset = init_test_subset[np.argsort(init_pehes)[-config.acquisition.n_test_samples:]]
    else:
        raise ValueError("Invalid subsample method")
    
    logging.info(f"Running the training with the following acquisition metric: {config.acquisition_metric}")

    # run the acquisition loops
    pehe_over_time = np.zeros((len(test_subset), (d - num_ones)))
    pehe_final = []
    actions_list = []
    number_acquisitions_list = []
    score_list = []
    stype_error_over_time = np.zeros((len(test_subset), (d - num_ones)))
    tau_var_over_time = np.zeros((len(test_subset), (d - num_ones)))
    if config.dataset.dataset_name == 'acic2016':
        if config.acquisition.track_predictive_acquisition:
            predictive_over_time = np.zeros((len(test_subset), (d - num_ones)))
        if config.dataset.dataset.setup_pi == 'overlap_violation':
            overlap_over_time = np.zeros((len(test_subset), (d - num_ones)))

    for k in range(len(test_subset)):
        start_time = time.time()
        i = test_subset[k]
        logging.info(f"Running acquisition for individual {k}. pehe with all the features acquired: {(tau_pred_full[i] - tau_test[i])**2}")
        unobserved_indices = np.argwhere(z_test[i] == 0).flatten()
        acquired_j = []
        pehe_per_step = []
        stype_error_per_step = []
        overlap_per_step = []
        tau_var_per_step = []
        action = 'withhold'
        if config.dataset.dataset_name == 'acic2016':
            if config.acquisition.track_predictive_acquisition:
                predictive_per_step = []
        
        # evaluating the model before the acquisition
        X_torch = torch.from_numpy(x_test[i]).float().unsqueeze(0)
        Z_torch = torch.from_numpy(z_test[i]).float()
        new_x_samples = gsm.sample_conditional_covariates(X_torch, Z_torch, n_samples=config.acquisition.n_cond_samples, seed=config.random_seed)
        mu0_full_samples, mu1_full_samples = model.predict_mus_from_covariates(new_x_samples, posterior_sample=config.acquisition.n_posterior_samples,
                                                                        batch_size=16384)
        mu0_full_samples = mu0_full_samples[0]
        mu1_full_samples = mu1_full_samples[0]

        while len(unobserved_indices) > 0:
            if config.acquisition_metric == 'random':
                j_to_acquire = np.random.choice(unobserved_indices)
            
            else:
                results = []
                for j in unobserved_indices:
                    samples_xj, z_new_j = gsm.sample_single_covariate(x_test[i], z_test[i], j, n_samples=config.acquisition.n_j_samples, seed=config.random_seed)
                    samples_conditional_x = gsm.sample_conditional_covariates(samples_xj, z_new_j, n_samples=config.acquisition.n_cond_samples, seed=config.random_seed)
                    mu0_samples, mu1_samples = model.predict_mus_from_covariates(samples_conditional_x, posterior_sample=config.acquisition.n_posterior_samples,
                                                                                 batch_size=16384)
                    
                    results.append(acquisition_metric(mu0_samples, mu1_samples))

                j_to_acquire = unobserved_indices[np.argmax(results)]
            
            acquired_j.append(j_to_acquire)
            z_test[i, j_to_acquire] = 1
            unobserved_indices = np.argwhere(z_test[i] == 0).flatten()

            # evaluating the model after the acquisition
            X_torch = torch.from_numpy(x_test[i]).float().unsqueeze(0)
            Z_torch = torch.from_numpy(z_test[i]).float()
            new_x_samples = gsm.sample_conditional_covariates(X_torch, Z_torch, n_samples=config.acquisition.n_cond_samples, seed=config.random_seed)
            mu0_full_samples, mu1_full_samples = model.predict_mus_from_covariates(new_x_samples, posterior_sample=config.acquisition.n_posterior_samples,
                                                                         batch_size=16384)
            mu0_full_samples = mu0_full_samples[0]
            mu1_full_samples = mu1_full_samples[0]
            tau_pred = (mu1_full_samples - mu0_full_samples).mean()
            tau_pred_var = (mu1_full_samples - mu0_full_samples).var(1).mean() + (mu1_full_samples - mu0_full_samples).mean(1).var()
            pehe = (tau_pred - tau_test[i])**2
            stype_error = int((tau_test[i] > 0) != (tau_pred > 0))
            if config.dataset.dataset_name == 'acic2016' and config.dataset.dataset.setup_pi == 'overlap_violation':
                logging.info(f"Acquiring feature {j_to_acquire}. Was this feature predictive? {"Yes." if np.isin(j_to_acquire, ds_test.predictive) else "No."} Was this feature related to overlap? {"Yes." if np.isin(j_to_acquire, ds_test.overlap) else "No."} New pehe: {pehe}.")
            elif config._dataset.dataset_name == 'acic2016':
                logging.info(f"Acquiring feature {j_to_acquire}. Was this feature predictive? {"Yes." if np.isin(j_to_acquire, ds_test.predictive) else "No."} New pehe: {pehe}.")
            else:
                logging.info(f"Acquiring feature {j_to_acquire}. New pehe: {pehe}.")
            pehe_per_step.append(pehe)
            stype_error_per_step.append(stype_error)
            tau_var_per_step.append(tau_pred_var)

            # track the proportion of the predictive features acquired
            if config.dataset.dataset_name == 'acic2016':
                if config.acquisition.track_predictive_acquisition:
                    acquired = np.argwhere(z_test[i] == 1).flatten()
                    prop_predictive_acquired = len(set(acquired).intersection(set(ds_test.predictive)))/len(ds_test.predictive)
                    predictive_per_step.append(prop_predictive_acquired)
                if config.dataset.dataset.setup_pi == 'overlap_violation':
                    prop_overlap_acquired = len(set(acquired).intersection(set(ds_test.overlap)))/len(ds_test.overlap)
                    overlap_per_step.append(prop_overlap_acquired)

            # check whether to continue acquisition
            if config.acquisition.threshold_variable == "TE":
                variance_dist = np.var(mu1_full_samples - mu0_full_samples, axis=1)
                pred_var = tau_pred_var
                stop_condition = pred_var < var_threshold
                overlap_violation = np.mean(variance_dist > var_threshold) > config.acquisition.proportion_above_threshold
            elif config.acquisition.threshold_variable == "PO":
                variance_dist_1 = mu1_full_samples.var(axis=1)
                variance_dist_0 = mu0_full_samples.var(axis=1)
                pred_var_1 = mu1_full_samples.var(1).mean() + mu1_full_samples.mean(1).var()
                pred_var_0 = mu0_full_samples.var(1).mean() + mu0_full_samples.mean(1).var()
                pred_var = np.max([pred_var_1, pred_var_0])
                stop_condition = pred_var < var_threshold
                overlap_violation = np.mean(variance_dist_1 > var_threshold) > config.acquisition.proportion_above_threshold or np.mean(variance_dist_0 > var_threshold) > config.acquisition.proportion_above_threshold
            elif config.acquisition.threshold_variable == "sTE":
                variance_dist = np.var(mu1_full_samples - mu0_full_samples > 0, axis=1)
                pred_var = (mu1_full_samples - mu0_full_samples > 0).var(1).mean() + (mu1_full_samples - mu0_full_samples > 0).mean(1).var()
                stop_condition = pred_var < var_threshold
                overlap_violation = np.mean(variance_dist > var_threshold) > config.acquisition.proportion_above_threshold
            logging.info(f"Predictive variance: {pred_var}.")
            if stop_condition:
                # stop acquisition and make recommendation if the variance is low
                action = "predict"
                logging.info(f"Acquisition stopped with action: {action} after {len(acquired_j)} steps.")
                break
            elif overlap_violation:
                # stop acquisition and withhold recommendation if there is little chance of lowering the variance
                action = 'withhold'
                logging.info(f"Acquisition stopped with action: {action} after {len(acquired_j)} steps.")
                break

        logging.info(f"trajectory per individual: {time.time() - start_time}")

        total_number_steps = len(pehe_per_step)
        pehe_over_time[k, :total_number_steps] = pehe_per_step
        stype_error_over_time[k, :total_number_steps] = stype_error_per_step
        tau_var_over_time[k, :total_number_steps] = tau_var_per_step
        if config.dataset.dataset_name == 'acic2016':
            if config.acquisition.track_predictive_acquisition:
                predictive_over_time[k, :total_number_steps] = predictive_per_step
            if config.dataset.dataset.setup_pi == 'overlap_violation':
                overlap_over_time[k, :total_number_steps] = overlap_per_step
        
        # decision making metrics
        actions_list.append(action)
        pehe_final.append(pehe_per_step[-1])
        number_acquisitions_list.append(total_number_steps)
        score = (0 if action == 'withhold' else (1 - 2*stype_error_per_step[-1]))
        score_list.append(score)
    
    # track the pehes
    pehe_over_time = np.sqrt(np.mean(pehe_over_time, axis=0))
    logging.info(f"Final pehes: {pehe_over_time}")

    # track the stype errors
    stype_error_over_time = np.mean(stype_error_over_time, axis=0)
    logging.info(f"Final stype errors: {stype_error_over_time}")

    # track the tau variance
    tau_var_over_time = np.mean(tau_var_over_time, axis=0)
    logging.info(f"Final tau variance: {tau_var_over_time}")

    if config.wandb_log:
        for timestep in range(len(pehe_over_time)):
            wandb.log({"timestep": timestep, "pehe": pehe_over_time[timestep], "stype_error": stype_error_over_time[timestep],
                       "tau_var": tau_var_over_time[timestep]})

    # track the acquisition of predictive features
    if config.dataset.dataset_name == 'acic2016':
        if config.acquisition.track_predictive_acquisition:
            predictive_over_time = np.mean(predictive_over_time, axis=0)
            logging.info(f"Final predictive acquisition: {predictive_over_time}")
            if config.wandb_log:
                for timestep, pred in enumerate(predictive_over_time):
                    wandb.log({"timestep": timestep, "predictive_acquisition": pred})
        if config.dataset.dataset.setup_pi == 'overlap_violation':
            overlap_over_time = np.mean(overlap_over_time, axis=0)
            logging.info(f"Final overlap acquisition: {overlap_over_time}")
            if config.wandb_log:
                for timestep, overlap in enumerate(overlap_over_time):
                    wandb.log({"timestep": timestep, "overlap_acquisition": overlap_over_time[timestep]})
    
    # track decision making actions
    if config.wandb_log:
        wandb.log({"total_score": np.sum(np.array(score_list)), "total_cost": np.sum(np.array(number_acquisitions_list)),
                   "average_cost": np.mean(np.array(number_acquisitions_list)),
                   "number_withhold": np.sum(np.array(actions_list) == 'withhold'),
                   "number_predict": np.sum(np.array(actions_list) == 'predict'),
                   "average_cost_for_withhold": np.mean(np.array(number_acquisitions_list)[np.array(actions_list) == 'withhold']),
                   "average_cost_for_predict": np.mean(np.array(number_acquisitions_list)[np.array(actions_list) == 'predict']),
                   "average_score_for_predict": np.mean(np.array(score_list)[np.array(actions_list) == 'predict']),
                   "pehe_for_predict": np.sqrt(np.mean(np.array(pehe_final)[np.array(actions_list) == 'predict'])),
                   "pehe_for_withhold": np.sqrt(np.mean(np.array(pehe_final)[np.array(actions_list) == 'withhold'])),
                   "total_score_per_cost": np.sum(np.array(score_list))/np.sum(np.array(number_acquisitions_list))})

    # save the results
    if config.save_to_csv:
        pehes_dict = {f"t{i}": pehe for i, pehe in enumerate(pehe_over_time)}
        pehes_dict['acquisition_metric'] = config.acquisition_metric
        if (experiment_dir / "results.csv").exists():
            results_df = pd.read_csv(experiment_dir / "results.csv", index_col=False)
            new_df = pd.DataFrame([pehes_dict])
            results_df = pd.concat([results_df, new_df], ignore_index=True)
        else:
            results_df = pd.DataFrame([pehes_dict])
        results_df.to_csv(experiment_dir / "results.csv", index=False)
    
    print("yaaay!")

if __name__ == "__main__":
    train()