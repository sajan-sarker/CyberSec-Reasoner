import logging
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)

def load_tokenizer(model_cfg, dataset_cfg):
    """ Load the tokenizer """
    model_name = model_cfg['name']

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg["trust_remote_code"],
        padding_side=dataset_cfg["padding_side"],
        truncation_side=dataset_cfg["truncation_side"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("pad_token was None — set to eos_token.")
    logger.info(f"Tokenizer loaded:  {model_name}")
    logger.info(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    logger.info(f"  BOS: {tokenizer.bos_token!r}  EOS: {tokenizer.eos_token!r}  PAD: {tokenizer.pad_token!r}")
    return tokenizer

def load_base_model(cfg):
    """ Load the base model """
    model_name = cfg['name']
    
    logger.info(f"Loading base model: {model_name} with torch_dtype={cfg['torch_dtype']} and attn_implementation={cfg['attn_implementation']}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["torch_dtype"]),
        attn_implementation=cfg['attn_implementation'],
        trust_remote_code=cfg["trust_remote_code"],
        device_map=cfg["device_map"]
    )

    # Enable gradient checkpointing for memory efficiency during training
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model.config.use_cache = False  # Disable caching for training

    logger.info(f"Based model {model_name} loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")
    logger.info(f"Model Loaded on device: {next(model.parameters()).device}")
    return model

def load_qlora_base_model(cfg):
    """ Load the QLoRA model """
    # Configure bitsandbytes for 8-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=cfg["load_in_8bit"],
        llm_int8_threshold=cfg["llm_int8_threshold"],
        llm_int_skip_modules=cfg["llm_int_skip_modules"],
        llm_int8_enable_fp32_cpu_offload=cfg["llm_int8_enable_fp32_cpu_offload"]
    )

    logger.info(f"Loading QLoRA model: {cfg['name']} with bitsandbytes config: {bnb_config}")

    # Load the model in 8-bit precision using bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, cfg["torch_dtype"]),
        attn_implementation=cfg['attn_implementation'],
        trust_remote_code=cfg["trust_remote_code"],
        device_map=cfg["device_map"]
    )

    # enable gradient checkpointing for memory efficiency during training
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model.config.use_cache = False  # disable caching for training

    logger.info(f"QLoRA model {cfg['name']} loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")
    logger.info(f"Model Loaded on device: {next(model.parameters()).device}")
    return model

def create_lora_config(cfg):
    """ Create the LoRA config """
    config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        task_type=TaskType[cfg["task_type"]],
        use_rslora=cfg["use_rslora"]
    )

    logger.info(f"LoRA config created with r={cfg['r']}, alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']}, "
                f"bias={cfg['bias']}, task_type={cfg['task_type']}, use_rslora={cfg['use_rslora']},"
                f"\ntarget_modules={cfg['target_modules']}, ")
    
    return config