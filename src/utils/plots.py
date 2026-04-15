import os
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_learning_curve(trainer, msg="Learning Curve", save_dir="./plots"):
    """ Plots the training and validation loss curves from the trainer's log history."""

    train_loss = []
    train_steps = []
    val_loss = []
    val_steps = []
    path = f"{save_dir}/{msg}.png"
    for log in trainer.state.log_history:
        if 'loss' in log:
            train_loss.append(log['loss'])
            train_steps.append(log['step'])
        if 'eval_loss' in log:
            val_loss.append(log['eval_loss'])
            val_steps.append(log['step'])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Training Losses")
    plt.plot(val_steps, val_loss, label="Validation Losses")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"{msg}")
    plt.legend()
    plt.savefig(path)
    plt.close()
    logger.info(f"{msg} saved at: {path}")

def plot_metrics(results, msg="Evaluation Metrics", save_dir="./plots"):
    """ Plots the evaluation metrics as a bar chart. """
    metrics = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, scores, color=['blue', 'orange', 'green'])
    plt.title(f"{msg}")
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.ylim(0, max(scores) * 1.2)
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f"{score:.4f}", ha='center', va='bottom')
    plt.legend()
    plt.savefig(f"{save_dir}/{msg}.png")
    plt.close()
    logger.info(f"{msg} saved at: {save_dir}/{msg}.png")

def plot_grpo_metrics(results, msg="GRPO Model Evaluation Metrics", save_dir="./plots"):
    """ Plots the evaluation metrics as a bar chart. """
    filtered_results = {k: v for k, v in results.items() if k != "eval/num_samples"}
    metrics = [k.replace("eval/", "") for k in filtered_results.keys()]
    metrics = [k.replace("eval/", "").replace("_", " ").title() for k in filtered_results.keys()]
    scores = [v * 100 for v in filtered_results.values()]

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, scores)
    plt.title(msg)
    plt.xlabel('Metrics')
    plt.ylabel('Scores (%)')
    plt.ylim(0, 100)

    for i, score in enumerate(scores):
        plt.text(i, score + 1, f"{score:.2f}", ha='center', va='bottom')

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{msg}.png")
    plt.savefig(file_path)
    plt.close()
    logger.info(f"{msg} saved at: {save_dir}/{msg}.png")