# pip install tensorflow

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Data
x = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)


# WITH KERAS
"""
# model
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
# Compile
model.compile(optimizer='sgd', loss='mean_squared_error')
#train
model.fit(x,y, epochs=3000)
#predict 
print(model.predict(np.array([7], dtype=float)))
"""

# WITH OUT KERAS

# Initialize Bias and Weight
W = tf.Variable(0.5)
b = tf.Variable(0.0)

# Prefiction (Forward Pass)
def predict(x):
    return W * x + b

# Define Loss Function
def loss_fun(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Optimizer
optimizer = tf.optimizers.SGD(learning_rate = 0.01)


# Training Loop
for epoch in range(500):
    with tf.GradientTape() as tape:
        y_pred = predict(x)
        loss = loss_fun(y, y_pred)

    # Compute Gradients of W and b
    gradients = tape.gradient(loss, [W, b])

    # Update Weight
    optimizer.apply_gradients(zip(gradients, [W, b]))



test_x = 7.0

predted_Y = predict(test_x)
print(predted_Y)




