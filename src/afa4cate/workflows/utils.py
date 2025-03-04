from pathlib import Path

def get_experiment_dir(config, job_dir):
    if config.dataset_name == "acic2016":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"ns-{config.dataset.n_samples}_sp-{config.dataset.setup_pi}_rte-{config.dataset.rho_TE}_s-{config.dataset.seed}"
        )
    elif config.dataset_name == "ihdp":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"s-{config.dataset.seed}"
        )
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}")


def get_tuning_dir(config, job_dir):
    if config.dataset_name == "acic2016":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"ns-{config.dataset.n_samples}_sp-{config.dataset.setup_pi}_rte-{config.dataset.rho_TE}_s-{config.dataset.seed}"
        )
    elif config.dataset_name == "ihdp":
        return (
            Path(job_dir)
            / config.dataset_name
            / f"s-{config.dataset.seed}"
        )
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}")