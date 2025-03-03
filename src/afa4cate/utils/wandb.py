from omegaconf import DictConfig

import wandb

from afa4cate.utils.configs import flatten_config


def initialize_wandb(cfg: DictConfig) -> str | None:
    """Initialize wandb."""
    cfg_flat = flatten_config(cfg)
    wandb.init(project="AFA4CATE_FINAL", config=cfg_flat, tags=["afa4cate"])
    assert wandb.run is not None
    run_id = wandb.run.id
    assert isinstance(run_id, str)
    return run_id
