from pathlib import Path

def get_experiment_dir(config, job_dir):
    if config.dataset_name == "synthetic":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"ns-{config.dataset.n_samples}_sm-{config.dataset.setup_mu}_sp-{config.dataset.setup_pi}_d-{config.dataset.d}_rcov-{config.dataset.rho_cov}_rte-{config.dataset.rho_TE}_l-{config.dataset.lambd}_s-{config.dataset.seed}"
        )
    elif config.dataset_name == "ihdp":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"s-{config.dataset.seed}"
        )


def get_tuning_dir(config, job_dir):
    if config.dataset_name == "synthetic":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"ns-{config.dataset.n_samples}_sm-{config.dataset.setup_mu}_sp-{config.dataset.setup_pi}_d-{config.dataset.d}_rcov-{config.dataset.rho_cov}_rte-{config.dataset.rho_TE}_l-{config.dataset.lambd}"
        )
    elif config.dataset_name == "ihdp":
        return (
            Path(job_dir)
            / config.dataset_name
        )