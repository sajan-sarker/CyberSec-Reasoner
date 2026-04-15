import os
import math
import json
import torch
import random
import logging
import warnings
import evaluate
import argparse
from pathlib import Path

# set current working directory
os.chdir(Path.cwd())

from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.logger import setup_logging
from src.config_loader import load_config
from src.utils.utils import setup_env, log_gpu_info
from src.utils.seed import setup_seed
from src.utils.wandb import init_wandb, finish_wandb
from src.model_loader import load_base_model_for_evaluation, load_tokenizer
from src.data_loader import split_prompt_and_target
from src.grpo_evaluation_fnc import cwe_accuracy, formatting_accuracy, keyword_alignment_accuracy
from src.utils.plots import plot_grpo_metrics

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

logger.info(f"Current Working directory set to: {os.getcwd()}")

def generate_batch(model, tokenizer, prompts, eval_cfg, device):
    """Tokenize a batch of prompts and return decoded completions."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=eval_cfg["max_length"],
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=eval_cfg["max_new_tokens"],
            temperature=eval_cfg["temperature"],
            top_p=eval_cfg["top_p"],
            repetition_penalty=eval_cfg["repetition_penalty"],
            do_sample=eval_cfg["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens — keep only newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:]
    completions = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    return completions

def main(args):
    config_path = args.config_path
    ## setup logging
    setup_logging()
    
    model_cfg, dataset_cfg, _, training_cfg, wandb_cfg, paths_cfg, eval_cfg = load_config(config_path)

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
    tokenizer.padding_side = "left"

    # dataset
    dataset = load_from_disk(eval_cfg["data_path"])    
    dataset = dataset["test"]
    logger.info(f"Dataset stats: {dataset}")

    # load model for generation
    log_gpu_info("Before loading model")
    model = load_base_model_for_evaluation(eval_cfg)
    model.eval()
    log_gpu_info("After loading model")

    batch_size = eval_cfg["batch_size"]
    generations_file = Path(eval_cfg["generations_file"])
    generations_file.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    records = []
    scores = {
        "cwe_accuracy": [],
        "formatting_accuracy": [],
        "keyword_alignment_accuracy": []
    }

    logger.info(f"Starting evaluation on {len(dataset)} samples (batch_size={batch_size})…")
    
    # evaluation loop
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(batch_start, min(batch_start + batch_size, len(dataset))))

        prompts = []
        references = []

        for sample in batch:
            prompt, reference = split_prompt_and_target(sample, tokenizer)
            prompts.append(prompt)
            references.append(reference)

        completions = generate_batch(model, tokenizer, prompts, eval_cfg, device)

        for prompt, completion, reference in zip(prompts, completions, references):
            cwe_acc  = cwe_accuracy(completion, reference)
            fmt_acc  = formatting_accuracy(completion)
            kw_acc   = keyword_alignment_accuracy(completion)

            scores["cwe_accuracy"].append(cwe_acc)
            scores["formatting_accuracy"].append(fmt_acc)
            scores["keyword_alignment_accuracy"].append(kw_acc)

            records.append({
                "prompt": prompt,
                "completion": completion,
                "reference_cwe": reference,
                "cwe_accuracy": cwe_acc,
                "formatting_accuracy": fmt_acc,
                "keyword_alignment_accuracy": kw_acc
            })

    # aggregate results
    n = len(records)
    results = {
        "eval/cwe_accuracy": sum(scores["cwe_accuracy"]) / n,
        "eval/formatting_accuracy": sum(scores["formatting_accuracy"]) / n,
        "eval/keyword_alignment_accuracy": sum(scores["keyword_alignment_accuracy"]) / n,
        "eval/num_samples": n,
    }

    logger.info("Evaluation Results...")
    for metric, score in results.items():
        logger.info(f"{metric}: {score:.4f}")

    wandb_run.log(results)

    with open(generations_file, "w") as f:
        json.dump(records, f, indent=4)
    logger.info(f"Generations and evaluation records saved to {generations_file}")

    plot_grpo_metrics(results, "GRPO Model Evaluation Results")

    # print results
    logger.info("Evaluation Results:")
    for metric, score in results.items():
        logger.info(f"{metric}: {score:.4f}")

    finish_wandb(wandb_run)
    logger.info("Evaluation Complete.\nWandb run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take config path",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/grpo_qwen3.5_4b.yaml",
        help="Path to the training configuration file",
    )
    args = parser.parse_args()
    main(args)