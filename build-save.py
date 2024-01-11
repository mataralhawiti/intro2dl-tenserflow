import tensorflow as tf
import numpy as np
import logging

keras = tf.keras ## https://stackoverflow.com/questions/68860879/vscode-keras-intellisensesuggestion-not-working-properly

# set up traning data
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q) :
    print("{} degree celsius = {} degree fahrenheit".format(c, fahrenheit_a[i]))

# Create the model
"""
# not comman to define layers beforehands

layer0 = keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.models.Sequential(layer0)
"""
# similar to - it's common 
model = keras.models.Sequential(
    keras.layers.Dense(units=1, input_shape=[1]) # Just your regular densely-connected NN layer.
)

# Compile the model, with loss and optimizer functions
"""
In fact, the act of calculating the current loss of a model and then improving it is precisely **what training is**.

During training, the optimizer function is used to calculate adjustments to the model's internal variables.
The goal is to adjust the internal variables until **the model (which is really a math function)** mirrors the actual equation for converting Celsius to Fahrenheit.
"""
model.compile(loss="mean_squared_error", 
              optimizer=keras.optimizers.Adam(0.1)
              )


# Train the model
history = model.fit(x=celsius_q,
                    y=fahrenheit_a,
                    epochs=500
                )
print("Finished traing the model")

# evaluate the model
accuracy = model.evaluate(x=celsius_q, y=fahrenheit_a)
print('Accuracy: ', accuracy)


## use the Model
print(model.predict([100.0]))

"""
# Keras separates the concerns of :
- saving your model architecture
- saving your model weights.


Model weights are saved to an HDF5 format. This grid format is ideal for storing multi-dimensional arrays of numbers.

The model structure can be described and saved using two different formats: JSON and YAML.


Save Model to JSON
Save Model to YAML
Save Model to HDF5

The first two examples save the model architecture and weights separately. The model weights are saved into an HDF5 format file in all cases.
"""

# serialize model to JSON
model_json = model.to_json()
with open("matar-model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("matar-model/model.h5")
print("Saved model to disk")


# later...
# load json and create model and use it