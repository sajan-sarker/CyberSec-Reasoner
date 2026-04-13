import os
import sys
import torch
import random
import argparse
import logging
import warnings
from pathlib import Path

# set current working directory
os.chdir(Path.cwd())

from datasets import DatasetDict
from src.utils.logger import setup_logging
from src.config_loader import load_config
from src.utils.utils import setup_env, log_gpu_info
from src.utils.seed import setup_seed
from src.utils.wandb import init_wandb, finish_wandb
from src.model_loader import load_tokenizer, load_qlora_base_model
from src.data_loader import load_dataset, apply_chat_tempalte
from src.dataset_stats import print_token_stats
from src.trainer import build_sft_config, build_trainer, save_adapter, merge_and_save
from src.utils.plots import plot_learning_curve

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

logger.info("Current Working directory set to: {os.getcwd()}")

def main(args):
    train_type = args.train_type
    config_path = args.config_path
    if train_type not in ["soft", "full"]:
        print(f"Error: Invalid training type '{type}'. Must be 'soft' or 'full'.")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    ## setup logging
    setup_logging()
    
    model_cfg, dataset_cfg, lora_config, training_cfg, wandb_cfg, paths_cfg, _ = load_config(config_path)
    
    # setup environment
    setup_env()
    
    # log GPU info
    log_gpu_info()
    
    # setup seed
    setup_seed(training_cfg)
    
    # init wandb
    wandb_run = init_wandb(training_cfg["report_to"], wandb_cfg)

    # tokenizer
    tokenizer = load_tokenizer(model_cfg, dataset_cfg)

    # dataset
    dataset = load_dataset(dataset_cfg)

    if train_type == "soft":
        # create smaller splits
        sample_dataset = DatasetDict({
            "train": dataset["train"].select(range(2000)),
            "validation": dataset["validation"].select(range(200)),
            "test": dataset["test"].select(range(200)),
        })
        dataset = sample_dataset
        print(dataset)
    
    # apply chat template
    dataset = apply_chat_tempalte(dataset, tokenizer, "qwen")
    
    print_token_stats(dataset, tokenizer)
    
    # load model
    log_gpu_info("Before loading model")
    model = load_qlora_base_model(model_cfg)
    log_gpu_info("After loading model")
    
    # init trainer
    sft_config = build_sft_config(dataset_cfg, training_cfg, wandb_cfg["run_name"])
    
    trainer = build_trainer(
        model, 
        tokenizer, 
        dataset,
        sft_config,
        lora_config
    )
    
    # start training
    log_gpu_info("Before training")
    logger.info("Starting training...")
    result = trainer.train()
    log_gpu_info("After training")
    logger.info(f"Training completed. Training result: {result}")
    
    metrics = result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if train_type == "soft":
        plot_learning_curve(trainer, "Qwen3.5 4B Soft SFT Learning Curve", paths_cfg["plots_dir"])
    else:
        plot_learning_curve(trainer, "Qwen3.5 4B SFT Learning Curve", paths_cfg["plots_dir"])
    
    # save
    save_adapter(trainer, tokenizer, paths_cfg)
    merge_and_save(model_cfg, paths_cfg)
    
    finish_wandb(
        wandb_run
    )
    logger.info("Wandb run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take training type and config path",
    )
    parser.add_argument(
        "--train_type", 
        type=str, 
        default="soft", 
        help="Type of training: soft (test script), full"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/sft_qwen3.5_4b_soft.yaml",
        help="Path to the training configuration file"
    )
    args = parser.parse_args()
    main(args)