import math
import torch
import logging

from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from src.model_loader import create_lora_config
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

def build_sft_config(dataset_cfg, training_cfg, run_name, evaluation=True):
    """ Builds the SFTConfig object from the provided configuration dictionary."""
    if evaluation!=True:
        logger.warning("Evaluation is disabled. Skipping evaluation during training")
        return SFTConfig(
            output_dir=training_cfg["output_dir"],
            logging_dir=training_cfg["logging_dir"],

            num_train_epochs=training_cfg["num_train_epochs"],
            per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],

            # optimizer
            learning_rate=training_cfg["learning_rate"],
            lr_scheduler_type=training_cfg["lr_scheduler_type"],
            warmup_ratio=training_cfg["warmup_ratio"],
            weight_decay=training_cfg["weight_decay"],
            max_grad_norm=training_cfg["max_grad_norm"],
            optim=training_cfg["optim"],

            # precision
            fp16=training_cfg["fp16"],
            bf16=training_cfg["bf16"],
            tf32=training_cfg["tf32"],

            # data loading
            dataloader_num_workers=training_cfg["dataloader_num_workers"],
            dataloader_pin_memory=training_cfg["dataloader_pin_memory"],
            remove_unused_columns=training_cfg["remove_unused_columns"],

            # logging
            logging_steps=training_cfg["logging_steps"],
            report_to=training_cfg["report_to"],
            run_name=run_name,

            # saving
            eval_strategy="no",
            save_strategy=training_cfg["save_strategy"],
            save_steps=training_cfg["save_steps"],
            save_total_limit=training_cfg["save_total_limit"],
            load_best_model_at_end=training_cfg["load_best_model_at_end"],

            seed=training_cfg["seed"],

            # sft specific
            max_length=dataset_cfg["max_length"],
            packing=dataset_cfg["packing"],
            assistant_only_loss=training_cfg["assistant_only_loss"],
            neftune_noise_alpha=training_cfg["neftune_noise_alpha"],
            dataset_text_field=training_cfg["dataset_text_field"]
        )

    return SFTConfig(
        output_dir=training_cfg["output_dir"],
        logging_dir=training_cfg["logging_dir"],

        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],

        # optimizer
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        optim=training_cfg["optim"],

        # precision
        fp16=training_cfg["fp16"],
        bf16=training_cfg["bf16"],
        tf32=training_cfg["tf32"],

        # data loading
        dataloader_num_workers=training_cfg["dataloader_num_workers"],
        dataloader_pin_memory=training_cfg["dataloader_pin_memory"],
        remove_unused_columns=training_cfg["remove_unused_columns"],

        # logging
        logging_steps=training_cfg["logging_steps"],
        report_to=training_cfg["report_to"],
        run_name=run_name,

        # evaluation and saving
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=training_cfg["eval_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=training_cfg["greater_is_better"],

        seed=training_cfg["seed"],

        # sft specific
        max_length=dataset_cfg["max_length"],
        packing=dataset_cfg["packing"],
        assistant_only_loss=training_cfg["assistant_only_loss"],
        neftune_noise_alpha=training_cfg["neftune_noise_alpha"],
        dataset_text_field=training_cfg["dataset_text_field"]
    )

def build_trainer(model, tokenizer, dataset, sft_config, lora_config, evaluation=True):
    """ Builds the SFTTrainer object with the provided model, tokenizer, dataset, and configurations."""
    logger.info("Create LoRA Configs...")
    lora_config = create_lora_config(lora_config)
    
    logger.info("Prepare the model for LoRA fine-tuning...")
    model = get_peft_model(model, lora_config)
    
    logger.info(f"Peft Model Info: {model.print_trainable_parameters()}")

    logger.info("Innitializing the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if evaluation else None,
        args=sft_config
    )

    # calculate total training steps manually
    num_samples        = len(dataset["train"])
    batch_size         = sft_config.per_device_train_batch_size
    grad_accum         = sft_config.gradient_accumulation_steps
    epochs             = sft_config.num_train_epochs
    steps_per_epoch    = math.ceil(num_samples / (batch_size * grad_accum))
    total_steps        = math.ceil(steps_per_epoch * epochs)

    logger.info("SFTTrainer initialized successfully.")
    logger.info(f"Number of trainable parameters    : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Number of non-trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    logger.info(f"Effective batch size (per_device * grad_accum): {batch_size * grad_accum}")
    logger.info(f"Steps per epoch                   : {steps_per_epoch}")
    logger.info(f"Total training steps              : {total_steps}")
    return trainer

def save_adapter(trainer, tokenizer, cfg):
    """ Save the trained LoRA adapter and tokenizer to the specified output directory."""
    tokenizer.save_pretrained(cfg['tokenizer_save_dir'])
    trainer.save_model(cfg['final_adapter_dir'])
    trainer.save_state()

    logger.info(f"LoRA adapter saved to: {cfg['final_adapter_dir']}")
    logger.info(f"Tokenizer saved to: {cfg['tokenizer_save_dir']}")

    # list saved files 
    for f in sorted(Path(cfg['final_adapter_dir']).glob("*")):
        if f.is_file():
            logger.info(f"   {str(f.relative_to(cfg['final_adapter_dir']))} {f.stat().st_size / 1e6:.2f} MB")

def merge_and_save(model_cfg, paths_cfg, dtype=torch.bfloat16):
    """ Merge the LoRA adapter with the base model and save the merged model to the specified output directory."""
    logger.info("Merging the LoRA adapter with the base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"]
    )

    merged = PeftModel.from_pretrained(
        base,
        paths_cfg["final_adapter_dir"],
    )
    merged = merged.merge_and_unload()  # merge the LoRA adapter into the base model and unload the adapter weights from memory

    merged.save_pretrained(paths_cfg["merged_model_dir"], safe_serialization=True)
    logger.info(f"Merged model saved to: {paths_cfg['merged_model_dir']}")