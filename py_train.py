import os
import tensorflow as tf
from config.env_setup import setup_env
from models.simple_cnn import SimpleCNN
from data.dataset import get_mnist_dataset
from training.strategy import init_strategy
from training.trainer import train_model

def main():
    # Setup distributed environment
    rank, master_addr, master_port = setup_env()
    strategy = init_strategy(rank, master_addr, master_port)

    # Bind each process to its GPU
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[local_rank], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[local_rank], True)
        except RuntimeError as e:
            print(f"RANK {rank}: GPU setup failed -> {e}")

    # Build model under strategy scope
    with strategy.scope():
        model = SimpleCNN()
        optimizer = tf.keras.optimizers.Adam(0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Load MNIST dataset
    train_dataset = get_mnist_dataset(batch_size=64)

    # Train
    train_model(strategy, model, optimizer, loss_fn, train_dataset, rank)

if __name__ == "__main__":
    main()
