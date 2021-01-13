import tensorflow as tf


class StdNormalization(tf.keras.layers.Layer):
	"""normalizing input feature to std Gauss (mean 0, var 1)"""
	def __init__(self, mean_x, std_x, name='Std_Normalize', **kwargs):
		kwargs.update({'name': name, 'trainable': False})
		super(StdNormalization, self).__init__(**kwargs)
		self.mean_x = mean_x
		self.std_x = std_x

	def get_config(self):
		config = super(StdNormalization, self).get_config()
		config.update({'mean_x': self.mean_x, 'std_x': self.std_x})
		return config

	def call(self, x):
		return (x - self.mean_x) / self.std_x


class StdUnnormalization(StdNormalization):
	""" rescaling feature to original domain """

	def __init__(self, mean_x, std_x, name='Un_Normalize', **kwargs):
		super(StdUnnormalization, self).__init__(mean_x=mean_x, std_x=std_x, name=name, **kwargs)

	def call(self, x):
		return (x * self.std_x) + self.mean_x


class MinMaxNormalization(tf.keras.layers.Layer):
	"""normalizing input feature to std Gauss (mean 0, var 1)"""
	def __init__(self, min_x, max_x, name='MinMax_Normalize', **kwargs):
		kwargs.update({'name': name, 'trainable': False})
		super(MinMaxNormalization, self).__init__(**kwargs)
		self.min_x = min_x
		self.max_x = max_x

	def get_config(self):
		config = super(MinMaxNormalization, self).get_config()
		config.update({'min_x': self.min_x, 'max_x': self.max_x})
		return config

	def call(self, x):
		return (x - self.min_x) / (self.max_x - self.min_x)


class MinMaxUnnormalization(MinMaxNormalization):
	""" rescaling feature to original domain """

	def __init__(self, min_x, max_x, name='MinMax_Un_Normalize', **kwargs):
		super(MinMaxUnnormalization, self).__init__(min_x=min_x, max_x=max_x, name=name, **kwargs)

	def call(self, x):
		return x * (self.max_x - self.min_x) + self.min_x


class QuantileRegression():

	def __init__(self, quantile, n_layers=5, n_nodes=20):
		self.quantile = quantile
		self.n_layers = n_layers
		self.n_nodes = n_nodes

	def quantile_loss( self ):
		def loss( target, pred ):
			alpha = 1 - self.quantile
			err = target - pred
			return tf.where(err>=0, alpha*err, (alpha-1)*err)
		return loss


	def build(self):
		self.inputs = tf.keras.Input(shape=(1,))
		x = self.inputs
		for _ in range(self.n_layers):
			x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer='lecun_normal', activation='selu')(x)
		self.output = tf.keras.layers.Dense(1)(x)
		#self.output = tf.math.asinh(x) # output scaled to std normal distribution => last activation: arc sin hyperbolicus
		model = tf.keras.Model(self.inputs, self.output)
		model.compile(loss=self.quantile_loss(), optimizer='Adam') # Adam(lr=1e-3) TODO: add learning rate
		model.summary()
		return model


class QuantileRegressionV2():

	def __init__(self, n_layers=5, n_nodes=20, **kwargs):
		# super(QuantileRegressionV2, self).__init__(name='QuantileRegressionV2', **kwargs)
		self.n_layers = n_layers
		self.n_nodes = n_nodes

	def make_model(self, x_min_max=(0.,1.), y_min_max=(0.,1.)):
		inputs = tf.keras.Input(shape=(1,))
		x = MinMaxNormalization(*x_min_max)(inputs) #StdNormalization(*x_mean_std, name='Normalize')(inputs)
		for _ in range(self.n_layers):
			x = tf.keras.layers.Dense(self.n_nodes, activation='tanh')(x)
		outputs_normalized = tf.keras.layers.Dense(1)(x)
		outputs = outputs_normalized #MinMaxUnnormalization(*y_min_max)(outputs_normalized) #StdUnnormalization(*y_mean_std, name='Un-Normalize')(outputs_normalized)
		model = tf.keras.Model(inputs, outputs)
		return model

# ******************************************** #
#			quantile regression loss 		   #
# ******************************************** #

# @tf.function
def quantile_loss(targets, predictions, quantile):
	alpha = 1 - quantile
	err = targets - predictions
	return tf.where(err>=0, alpha*err, (alpha-1)*err)


