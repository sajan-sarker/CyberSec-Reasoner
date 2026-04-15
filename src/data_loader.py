# data loading and preprocessing functions
import logging
from pathlib import Path
from datasets import load_from_disk

logger = logging.getLogger(__name__)

def load_dataset(cfg):
    """ Load the dataset splits from disk """
    path = Path(cfg.get("path"))
    if not path.exists():
        logger.error(f"Dataset path {path} does not exist.")
        raise FileNotFoundError(f"Dataset path {path} does not exist.")
    
    dataset = load_from_disk(path)
    logger.info(
        f"Dataset loaded from {path} with "
        f"{len(dataset['train'])} training, "
        f"{len(dataset['validation']) if 'validation' in dataset else 0} validation, "
        f"{len(dataset['test']) if 'test' in dataset else 0} test samples."
    )
    print("Dataset:", dataset)
    return dataset

def apply_chat_tempalte(dataset, tokenizer, model_family="qwen"):
    """ Apply the chat template to the dataset samples """

    logger.info(f"Applying chat template for model family: {model_family}")
    if model_family.lower() == "qwen":
        formatted_dataset = dataset.map(
            apply_qwen_chat_template,
            desc="Applying Qwen chat template",
            fn_kwargs={"tokenizer": tokenizer}
        )
    else:
        logger.warning(f"Model family {model_family} not recognized.")
        raise ValueError(f"Model family {model_family} not recognized.")
    
    logger.info("Chat template applied to dataset.")
    return formatted_dataset

def apply_qwen_chat_template(data, tokenizer, column="messages"):
    """ apply the Qwen chat template to the datasets splits """
    text = tokenizer.apply_chat_template(
        data[column],
        tokenize=False,  # Tokenize it later before feeding into the model
        add_generation_prompt=True
    )
    return {"text": text}

def format_for_grpo(data):
    """ Format the data for GRPO training - extract the prompt and answer from the messages """
    data = data['messages']
    prompt = [
        data[0],
        data[1],
    ]
    return {
        "prompt": prompt,
        "answer": data[-1]['content']
    }

def split_prompt_and_target(data, tokenizer):
    """Split prompt and target from messages for evaluation."""

    # correct: pass list of messages directly (not nested)
    prompt_messages = data["messages"][:-1]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    reference = data["messages"][-1]["content"]

    return prompt, reference