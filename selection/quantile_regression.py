import tensorflow as tf


# ******************************************** #
#			quantile regression loss 		   #
# ******************************************** #

def quantile_loss(quantile):
	@tf.function
    def loss(target, pred):
        err = target - pred
        return tf.where(err>=0, quantile*err, (quantile-1)*err)
    return loss

# ******************************************** #
#			quantile regression models 		   #
# ******************************************** #


class QuantileRegression():

	def __init__(self, quantile, n_layers=5, n_nodes=20):
		self.quantile = quantile
		self.n_layers = n_layers
		self.n_nodes = n_nodes

	def build(self):
		self.inputs = tf.keras.Input(shape=(1,))
		x = self.inputs
		for _ in range(self.n_layers):
			x = tf.keras.layers.Dense(self.n_nodes, activation='relu')(x)
		self.output = tf.keras.layers.Dense(1)(x)
		#self.output = tf.math.asinh(x) # output scaled to std normal distribution => last activation: arc sin hyperbolicus
		model = tf.keras.Model(self.inputs, self.output)
		model.compile(loss=quantile_loss(self.quantile), optimizer='adam') # Adam(lr=1e-3) TODO: add learning rate
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

