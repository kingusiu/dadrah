import tensorflow as tf
import numpy as np
#import setGPU

print('tensorflow version: ', tf.__version__)

class CustomLayer(tf.keras.layers.Layer):
	def __init__(self, mean_x, var_x, **kwargs):
		super(CustomLayer, self).__init__(**kwargs)
		self.mean_x = mean_x
		self.var_x = var_x

	def call(self, x):
		return (x - self.mean_x) / self.var_x

	def get_config(self):
		config = super(CustomLayer, self).get_config()
		config.update({'mean_x': self.mean_x, 'var_x': self.var_x})
		return config



def make_model():
	inputs = tf.keras.Input(shape=(3,))
	x = tf.keras.layers.Dense(5)(inputs)
	x = CustomLayer(2., 1.)(x)
	outputs = tf.keras.layers.Softmax()(x)
	return tf.keras.Model(inputs, outputs)

model = make_model()
model.save('tmp_model.h5')
loaded_model = tf.keras.models.load_model('tmp_model.h5', custom_objects={'CustomLayer': CustomLayer})
x = tf.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))

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



