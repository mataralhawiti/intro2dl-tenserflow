import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


# Build a machine learning model

## Build a tf.keras.Sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

"""
Sequential is useful for stacking layers where each layer has one input tensor and one output tensor.

Layers are functions (Flatten, Dense, Dropout, etc) with a known mathematical structure that can be reused and have trainable variables

Most TensorFlow models are composed of layers.

This model uses the Flatten, Dense, and Dropout layers.
"""

# For each example, the model returns a vector of logits or log-odds scores, one for each class.
predictions = model(x_train[:1]).numpy()
print(predictions)