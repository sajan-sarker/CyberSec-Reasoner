import os
import logging
import wandb

logger = logging.getLogger(__name__)

def init_wandb(report_to, cfg):
    if report_to != "wandb":
        logger.warning("report_to is not set to 'wandb', skipping WandB initialization.")
        return None
    
    wb = cfg.get("wandb", {})

    # upload model artifact at end of run
    if cfg["log_model"]:
        os.environ["WANDB_LOG_MODEL"] = str(cfg["log_model"])

    # init run with config
    run = wandb.init(
        project=cfg["project"],
        name=cfg["run_name"],
        entity=cfg["entity"],
        tags=cfg["tags"],
        notes=cfg["notes"],
        config=cfg, # log entire config dict for easy search/filter in WandB UI
        resume="allow" # safe to re-run without creating duplicate runs
    )
    
    logger.info(f"WandB run '{run.name}' initialized. Dashboard: {run.get_url()}")
    return run

def finish_wandb(run, extra_metrics=None):
    if run is None:
        return
    if extra_metrics:
        run.log(extra_metrics)
    run.finish()
    logger.info("WandB run finished and synced.")