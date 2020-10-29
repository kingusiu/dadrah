import tensorflow as tf


class FeatureNormalization(tf.keras.layers.Layer):
	"""normalizing input feature to std Gauss (mean 0, var 1)"""
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


class FeatureUnNormalization(FeatureNormalization):
	""" rescaling feature to original domain """
	def call(self, x):
		return (x * self.var_x) + self.mean_x


class QuantileRegression():

	def __init__(self, quantile, n_nodes=20):
		self.quantile = quantile
		self.n_nodes_per_layer = n_nodes

	def quantile_loss( self ):
		def loss( target, pred ):
			alpha = 1 - self.quantile
			err = target - pred
			return tf.where(err>=0, alpha*err, (alpha-1)*err)
		return loss


	def build(self):
		self.inputs = tf.keras.Input(shape=(1,))
		x = tf.keras.layers.Dense(self.n_nodes_per_layer, activation='relu')(self.inputs)
		x = tf.keras.layers.Dense(self.n_nodes_per_layer, activation='relu')(x)
		x = tf.keras.layers.Dense(self.n_nodes_per_layer, activation='relu')(x)
		x = tf.keras.layers.Dense(self.n_nodes_per_layer, activation='relu')(x)
		self.output = tf.keras.layers.Dense(1)(x)
		#self.output = tf.math.asinh(x) # output scaled to std normal distribution => last activation: arc sin hyperbolicus
		model = tf.keras.Model(self.inputs, self.output)
		model.compile(loss=self.quantile_loss(), optimizer='Adam') # Adam(lr=1e-4) TODO: add learning rate
		model.summary()
		return model


class QuantileRegressionV2():

	def __init__(self, n_layers=3, n_nodes=20):
		self.n_layers = n_layers
		self.n_nodes = n_nodes

	def make_model(self, x_mean_var=(0.,1.), y_mean_var=(0.,1.)):
		inputs = tf.keras.Input(shape=(1,))
		x = FeatureNormalization(*x_mean_var)(inputs)
		for _ in range(n_layers):
			x = tf.keras.layers.Dense(self.n_nodes, activation='relu')(x)
		outputs_normalized = tf.keras.layers.Dense(1)(x)
		outputs = FeatureUnNormalization(*y_mean_var)(outputs_normalized)
		model = tf.keras.Model(inputs, outputs)
		return model

	def make_quantile_loss(self, quantile):

		@tf.function
		def quantile_loss(targets, predictions):
			alpha = 1. - quantile
			err = targets - predictions
			return tf.where(err>=0, alpha*err, (alpha-1)*err)
		
		return quantile_loss


