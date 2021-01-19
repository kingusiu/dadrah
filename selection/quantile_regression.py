import tensorflow as tf
import vande.vae.layers as layers


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

	def __init__(self, quantile, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu'):
		self.quantile = quantile
		self.n_layers = n_layers
		self.n_nodes = n_nodes
		self.x_mu_std = x_mu_std
		self.optimizer = optimizer
		self.initializer = initializer
		self.activation = activation

	def build(self):
		inputs = tf.keras.Input(shape=(1,))
    	x = layers.StdNormalization(*self.x_mu_std)(inputs)
    	for _ in range(self.n_layers):
        	x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)
    	outputs = tf.keras.layers.Dense(1, kernel_initializer=self.initializer)(x)
		model = tf.keras.Model(inputs, output)
		model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
		model.summary()
		return model
