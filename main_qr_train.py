import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import tensorflow as tf
import numpy as np
#import setGPU
from importlib import reload
import dadrah.playground.test_gradient_tape as tegrta
import dadrah.selection.loss_strategy as ls
import dadrah.selection.discriminator as disc
import dadrah.selection.quantile_regression as qure

print(tf.__version__)

x = tf.random.uniform([3000, 1], minval=1, maxval=100, dtype=tf.float32)
y = tf.random.uniform([3000,1], minval=1, maxval=10, dtype=tf.float32)

quantile = 0.1
strategy = ls.combine_loss_min

discriminator = disc.QRDiscriminator(quantile=quantile, loss_strategy=strategy, epochs=10, n_nodes=30)
discriminator.fit(x, y)
print(discriminator.model.summary())

x_test = tf.random.uniform([300, 1], minval=1, maxval=100, dtype=tf.float32)

y_test = discriminator.predict(x_test)

weights = discriminator.model.get_weights()

discriminator.save('./my_new_model.h5')

new_discriminator = disc.QRDiscriminator(quantile=quantile, loss_strategy=strategy)
new_discriminator.load('./my_new_model.h5')
loaded_weights = new_discriminator.model.get_weights()

print(weights[0])
print(loaded_weights[0])

for i in range(len(weights)):
    assert np.allclose(weights[0], loaded_weights[0])

y_loaded = new_discriminator.predict(x_test)

assert np.allclose(y_test, y_loaded)

