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
from src.data_loader import load_dataset, format_for_grpo
from src.dataset_stats import check_answer_length
from src.reward_functions import hybrid_reward
from src.trainer import build_grpo_config, build_grpo_trainer, save_adapter, merge_and_save

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

logger.info("Current Working directory set to: {os.getcwd()}")

def main(args):
    train_type = args.train_type
    config_path = args.config_path
    
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
    
    # formatt the dataset for grpo training
    dataset = dataset["train"].map(
        format_for_grpo,
        remove_columns=dataset["train"].column_names,
    )
    
    # check answer lengths
    check_answer_length(dataset, tokenizer)
    
    # load model
    log_gpu_info("Before loading model")
    model = load_qlora_base_model(model_cfg)
    log_gpu_info("After loading model")
    
    # init trainer
    grpo_config = build_grpo_config(training_cfg, wandb_cfg["run_name"])
    
    trainer = build_grpo_trainer(
        model, 
        tokenizer, 
        dataset,
        grpo_config,
        lora_config,
        hybrid_reward
    )
    
    # start training
    log_gpu_info("Before training")
    logger.info("Starting training...")
    result = trainer.train()
    log_gpu_info("After training")
    logger.info(f"Training completed. Training result: {result}")
    
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
        default="full", 
        help="Type of training: soft (test script), full"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/grpo_qwen3.5_4b.yaml",
        help="Path to the training configuration file"
    )
    args = parser.parse_args()
    main(args)