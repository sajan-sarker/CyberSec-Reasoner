import logging
from transformers import set_seed

logger = logging.getLogger(__name__)

def setup_seed(cfg):
    """ Set Python, NumPy, and PyTorch seeds for reproducibility."""
    set_seed(int(cfg["seed"]))
    
    logger.info(f"Global seed set to {cfg['seed']}.")