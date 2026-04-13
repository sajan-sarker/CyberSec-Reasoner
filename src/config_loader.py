""" Loads the configuration file and provides access to its contents. """

import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path):
    """ Load and validate the YAML config file """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path,"r") as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"Config loaded from: {config_path}")

    create_directories(cfg)

    model = cfg["model"]
    dataset = cfg["dataset"]
    lora_config = cfg["lora_config"]
    training = cfg["training"]
    evaluation = cfg["evaluation"]  
    wandb_cfg = cfg["wandb"]
    paths_cfg = cfg["paths"]
    
    # log the loaded configs for debugging
    logger.info(f"Model configs: {model}")
    logger.info(f"Dataset configs: {dataset}")
    logger.info(f"LoRA configs: {lora_config}")
    logger.info(f"Training configs: {training}")
    logger.info(f"Evaluation configs: {evaluation}")
    logger.info(f"WandB configs: {wandb_cfg}")
    logger.info(f"Paths configs: {paths_cfg}")

    return model, dataset, lora_config, training, wandb_cfg, paths_cfg, evaluation

def create_directories(cfg):
    """ create output directories declared in config """
    dirs = [
        cfg['training']['output_dir'],
        cfg['training']['logging_dir'],
        cfg['evaluation']['output_dir'],
        cfg['paths']['final_adapter_dir'],
        cfg['paths']['merged_model_dir'],
        cfg['paths']['tokenizer_save_dir'],
        cfg['paths']['plots_dir']
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Output directories created (if they did not exist)")