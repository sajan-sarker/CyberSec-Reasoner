import os
import torch
import logging

logger = logging.getLogger(__name__)

def log_gpu_info(tag=""):
    """ Print GPU availability, name, and total VRAM. """
    if not torch.cuda.is_available():
        logger.warning("No GPU detected. Training will be slow.")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.2f} GB VRAM. {tag}")

def log_gpu_memory(tag=""):
    """ Log currently allocated and reserved GPU memory. """
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(
        f"[{tag}] GPU Memory - "
        f"Allocated: {allocated:.2f} GB, "
        f"Reserved: {reserved:.2f} GB, "
        f"Total: {total:.2f} GB"
    )

def setup_env():
    """ Set environment variables for optimal GPU performance. """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for matmul and cuDNN.")