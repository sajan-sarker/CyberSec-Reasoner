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