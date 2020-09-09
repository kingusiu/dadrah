import tensorflow as tf
import numpy as np
import setGPU


@tf.function
def quantile_loss(targets, predictions):
	alpha = 1.-0.1
	err = targets - predictions
	return tf.where(err>=0, alpha*err, (alpha-1)*err)

def make_model(n_nodes=20):
	inputs = tf.keras.Input(shape=(1,))
	x = tf.keras.layers.Dense(n_nodes, activation='relu')(inputs)
	x = tf.keras.layers.Dense(n_nodes, activation='relu')(x)
	x = tf.keras.layers.Dense(n_nodes, activation='relu')(x)
	output = tf.keras.layers.Dense(1)(x)
	model = tf.keras.Model(inputs, output)
	return model

model = make_model()
print(model.summary())

targets = np.arange(0.,10.)
predictions = np.random.random(size=10)*10
print(targets)
print(predictions)
print(targets-predictions)
print(quantile_loss(targets, predictions))
