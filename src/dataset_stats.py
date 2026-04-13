import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def print_token_stats(dataset, tokenizer, text_column="text"):
    """ Check for average, minimum, and maximum number of tokens from the dataset after applying for chat template. """
    for split in dataset.keys():
        lengths = []
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            tokens = tokenizer(example[text_column], add_special_tokens=True)["input_ids"]
            lengths.append(len(tokens))
        lengths = np.array(lengths)
        logger.info(f"[{split}] stats -> Min: {lengths.min()} | Avg: {lengths.mean():.2f} | Max: {lengths.max()}")