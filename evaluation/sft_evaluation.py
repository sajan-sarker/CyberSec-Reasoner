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
from datasets import DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.logger import setup_logging
from src.config_loader import load_config
from src.utils.utils import setup_env, log_gpu_info
from src.utils.seed import setup_seed
from src.utils.wandb import init_wandb, finish_wandb
from src.model_loader import load_base_model_for_evaluation, load_tokenizer
from src.data_loader import load_dataset, split_prompt_and_target
from src.utils.plots import plot_metrics

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

logger.info(f"Current Working directory set to: {os.getcwd()}")

def generate_batch(model, tokenizer, prompts, cfg):
    """Run batched generation."""

    # ensure everything is string
    prompts = [str(p) for p in prompts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["max_input_length"],
    ).to(model.device)

    with torch.no_grad():
        outputs_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
        )

    # slice off the prompt tokens to get only the generated part
    generated_ids = []

    for i in range(outputs_ids.size(0)):
        input_len = inputs["attention_mask"][i].sum().item()
        generated_ids.append(outputs_ids[i, input_len:])
    
    generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [g.strip() for g in generations]

def compute_perplexity_loss(model, tokenizer, text):
    """Calculate perplexity for a single full text (prompt + reference)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return math.exp(outputs.loss.item())

def main(args):
    config_path = args.config_path

    # setup logging
    setup_logging()

    model_cfg, dataset_cfg, lora_config, training_cfg, wandb_cfg, paths_cfg, eval_cfg = load_config(config_path)

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
    dataset = load_dataset(dataset_cfg)

    # load model for generation
    log_gpu_info("Before loading model")
    model = load_base_model_for_evaluation(eval_cfg)
    model.eval()
    log_gpu_info("After loading model")

    # metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    # evaluation loop
    all_predictions = []
    all_references = []
    all_prompts = []
    pp_scores = []

    batch_prompts = []
    batch_references = []

    def flush_batch(batch_prompts, batch_references):
        """Generate predictions for a batch and collect metrics."""
        generations = generate_batch(model, tokenizer, batch_prompts, eval_cfg)

        for idx, generation in enumerate(generations):
            reference = batch_references[idx].strip()

            all_predictions.append(generation)
            all_references.append(reference)
            all_prompts.append(batch_prompts[idx])

            full_text = batch_prompts[idx] + reference
            pp_scores.append(compute_perplexity_loss(model, tokenizer, full_text))

    logger.info("Starting evaluation loop...")
    for data in tqdm(dataset["test"], desc="Evaluating..."):
        prompt, reference = split_prompt_and_target(data, tokenizer)

        if prompt is None or reference is None:
            continue
            
        batch_prompts.append(prompt)
        batch_references.append(reference)

        if len(batch_prompts) == eval_cfg["batch_size"]:
            flush_batch(batch_prompts, batch_references)
            batch_prompts, batch_references = [], []

    # handle any remaining prompts
    if batch_prompts:
        flush_batch(batch_prompts, batch_references)

    logger.info("Evaluation loop completed.")

    logger.info("Computing ROUGE, BLEU, and Perplexity scores...")
    results = {}

    results.update(rouge.compute(predictions=all_predictions, references=all_references))

    bleu_scores = bleu.compute(
        predictions=all_predictions,
        references=[[ref] for ref in all_references],
    )
    results["bleu"] = bleu_scores["bleu"]

    results["perplexity"] = sum(pp_scores) / len(pp_scores)

    plot_metrics(results, "SFT Evaluation Results")

    # print results
    logger.info("Evaluation Results:")
    for metric, score in results.items():
        logger.info(f"{metric}: {score:.4f}")

    # save predictions, references, and prompts for error analysis
    error_analysis_data = [
        {"prompt": prompt, "prediction": pred, "reference": ref}
        for prompt, pred, ref in zip(all_prompts, all_predictions, all_references)
    ]

    os.makedirs(eval_cfg["output_dir"], exist_ok=True)
    output_path = os.path.join(eval_cfg["output_dir"], "error_analysis_data.json")
    with open(output_path, "w") as f:
        json.dump(error_analysis_data, f, indent=4)

    logger.info(f"Error analysis data saved to {output_path}")

    finish_wandb(wandb_run)
    logger.info("Evaluation Complete.\nWandb run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take config path",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/sft_qwen3.5_4b.yaml",
        help="Path to the training configuration file",
    )
    args = parser.parse_args()
    main(args)