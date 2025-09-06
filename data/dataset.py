import tensorflow as tf

def get_mnist_dataset(batch_size=64):
    """Download, preprocess, and return MNIST train dataset."""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Normalize and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]  # shape: (28, 28, 1)

    # Create tf.data pipeline
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return dataset
