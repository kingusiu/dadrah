import tensorflow as tf


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

