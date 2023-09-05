# source : https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb

import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set up training data
"""
supervised Machine Learning is all about figuring out an algorithm given a set of inputs and outputs.
The task : to create a model that can give the temperature in Fahrenheit when given the degrees in Celsius.
we create 2 list to train our model
"""
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q) :
    print("{} degree celsius = {} degree fahrenheit".format(c, fahrenheit_a[i]))

# Create the model
"""
We will use the simplest possible model we can, a Dense network.
Since the problem is straightforward, this network will require only a single layer, with a single neuron.
"""

## Build a lyer
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
"""
input_shape=[1] —   This specifies that the input to this layer is a single value.
                    That is, the shape is a one-dimensional array with one member. 
                    Since this is the first (and only) layer, that input shape is the input shape of the entire model.
                    The single value is a floating point number, representing degrees Celsius.

units=1         —   This specifies the number of neurons in the layer. 
                    The number of neurons defines how many internal variables the layer has to try to learn how to solve the problem (more later).
                    Since this is the final layer, it is also the size of the model's output — a single float value representing degrees Fahrenheit.
                    (In a multi-layered network, the size and shape of the layer would need to match the input_shape of the next layer.)

"""

## Assemble Layers into the model
""" 
Once layers are defined, they need to be assembled into a model.
The Sequential model definition takes a list of layers as an argument, specifying the calculation order from the input to the output.

Note: You will often see the layers defined inside the model definition, rather than beforehand:
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])
"""

model = tf.keras.models.Sequential(l0)


## Compile the model, with loss and optimizer functions
"""
Before traing, the model has to be compiled. When compiled for training, the model is given :
- Loss function : A way of measuring how far off predictions are from the desired outcome. (The measured difference is called the "loss".)
- Optimizer function : A way of adjusting internal values in order to reduce the loss.
"""

model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))

"""
These are used during training (model.fit(), below) to first calculate the loss at each point, and then improve it.
In fact, the act of calculating the current loss of a model and then improving it is precisely **what training is**.

During training, the optimizer function is used to calculate adjustments to the model's internal variables.
The goal is to adjust the internal variables until **the model (which is really a math function)** mirrors the actual equation for converting Celsius to Fahrenheit.

TensorFlow uses numerical analysis to perform this tuning, and all this complexity is hidden from you so we will not go into the details here.
What is useful to know about these parameters are:

The loss function (mean squared error) and the optimizer (Adam) used here are standard for simple models like this one,
but many others are available. It is not important to know how these specific functions work at this point.

One part of the Optimizer you may need to think about when building your own models is the learning rate (0.1 in the code above).
This is the step size taken when adjusting values in the model.
If the value is too small, it will take too many iterations to train the model.Too large, and accuracy goes down.
Finding a good value often involves some trial and error, but the range is usually within 0.001 (default), and 0.1
"""


## Train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished traing the model")

## Display training statistics
# import matplotlib.pyplot as plt
# plt.xlabel('Epoch Number')
# plt.ylabel("Loss Magnitude")
# plt.plot(history.history['loss'])
     

## use the Model
print(model.predict([100.0]))