import tensorflow as tf
import numpy as np
import setGPU


class FeatureNormalization(tf.keras.layers.Layer):
	"""docstring for FeatureNormalization"""
	def __init__(self, mean_x, var_x, **kwargs):
		kwargs.update({'trainable': False})
		super(FeatureNormalization, self).__init__(**kwargs)
		self.mean_x = mean_x
		self.var_x = var_x

	def get_config(self):
		config = super(FeatureNormalization, self).get_config()
		config.update({'mean_x': self.mean_x, 'var_x': self.var_x})
		return config

	def build(self, input_shape):
		pass

	def call(self, x):
		return (x - self.mean_x) / self.var_x




class QuantileRegressionTest():

	def __init__(self):
		pass

	def make_model(self, x_mean_var=(0.,1.), n_nodes=20):
		inputs = tf.keras.Input(shape=(1,))
		x = FeatureNormalization(*x_mean_var)(inputs)
		x = tf.keras.layers.Dense(n_nodes, activation='relu')(x)
		x = tf.keras.layers.Dense(n_nodes, activation='relu')(x)
		x = tf.keras.layers.Dense(n_nodes, activation='relu')(x)
		output = tf.keras.layers.Dense(1)(x)
		model = tf.keras.Model(inputs, output)
		return model


	def make_quantile_loss(self, mean_target, var_target):

		@tf.function
		def quantile_loss(targets, predictions):
			targets = (targets - mean_target) / var_target # scale targets to normal dist
			alpha = 1.-0.1
			err = targets - predictions
			return tf.where(err>=0, alpha*err, (alpha-1)*err)
		
		return quantile_loss



if __name__ == "__main__":

	targets = np.arange(0.,10.)
	predictions = np.random.random(size=10)*10
	print(targets)
	print(predictions)
	print(targets-predictions)

	quantile_loss = make_quantile_loss(np.mean(targets), np.var(targets))

	print(quantile_loss(targets, predictions))
	print(quantile_loss(targets, predictions))

	x = tf.random.uniform((100,))

	model = make_model(x_mean_var=(np.mean(x), np.var(x)))
	print(model.summary())

	model.save('tmp_model.h5')

	loaded_model = tf.keras.models.load_model('tmp_model.h5', custom_objects={'FeatureNormalization': FeatureNormalization})
	assert np.allclose(model.predict(x), loaded_model.predict(x))
