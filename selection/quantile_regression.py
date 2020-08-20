import tensorflow as tf


class QuantileRegression():

	def __init__(self, quantile, num_nodes=20):
		self.quantile = quantile
		self.num_nodes_per_layer = num_nodes

	def quantile_loss( self ):
		def loss( target, pred ):
			alpha = 1 - self.quantile
			err = target - pred
			return tf.where(err>=0, alpha*err, (alpha-1)*err)
		return loss


	def build(self):
		self.inputs = tf.keras.Input(shape=(1,))
		x = tf.keras.layers.Dense(self.num_nodes_per_layer, activation='relu')(self.inputs)
		x = tf.keras.layers.Dense(self.num_nodes_per_layer, activation='relu')(x)
		x = tf.keras.layers.Dense(self.num_nodes_per_layer, activation='relu')(x)
		x = tf.keras.layers.Dense(self.num_nodes_per_layer, activation='relu')(x)
		self.output = tf.keras.layers.Dense(1)(x)
		model = tf.keras.Model(self.inputs, self.output)
		model.compile(loss=self.quantile_loss(), optimizer='Adam') # Adam(lr=1e-4) TODO: add learning rate
		model.summary()
		return model

