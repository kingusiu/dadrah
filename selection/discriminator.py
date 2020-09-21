import numpy as np
import tensorflow as tf
import sklearn.ensemble as scikit
import dadrah.selection.quantile_regression as qr
import pofah.jet_sample as js


class Discriminator():

	def __init__(self, quantile, loss_strategy):
		self.loss_strategy = loss_strategy
		self.quantile = quantile
		self.mjj_key = 'mJJ'

	def scale_output(self, outp):
		return (outp - self.mean_outp) / self.var_outp

	def unscale_output(self, outp):
		return (outp * self.var_outp) + self.mean_outp

	def fit(self, jet_sample):
		pass

	def save(self, path):
		pass

	def load(self, path):
		pass 

	def predict(self, data):
		'''predict cut for each example in data'''
		pass

	def select(self, jet_sample):
		pass

	def __repr__(self):
		return '{}% qnt, {} strategy'.format(str(self.quantile*100), self.loss_strategy.title_str)


class FlatCutDiscriminator(Discriminator):

	def fit(self, jet_sample):
		loss = self.loss_strategy(jet_sample)
		self.cut = np.percentile( loss, (1.-self.quantile)*100 )
		
	def predict(self, jet_sample):
		return np.asarray([self.cut]*len(jet_sample))

	def select(self, jet_sample):
		loss = self.loss_strategy(jet_sample)
		return loss > self.cut

	def __repr__(self):
		return 'Flat Cut: ' + Discriminator.__repr__(self)


class QRDiscriminator(Discriminator):

	def __init__(self, quantile, loss_strategy, batch_sz=128, epochs=100, **model_params):
		Discriminator.__init__(self, quantile, loss_strategy)
		self.batch_sz = batch_sz
		self.epochs = epochs
		self.model_params = model_params

	def training_step(self, step, x_batch, y_batch):
		# Open a GradientTape to record the operations run in forward pass
		with tf.GradientTape() as tape:
			predictions = self.model(x_batch, training=True)
			loss_value = self.loss_function(y_batch, predictions)

		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
		if step % 2 == 0:
			print("Training loss (for one batch) at step {}: {}".format(step, np.sum(loss_value)))


	def fit(self, x, loss):

		# process the input
		#x = jet_sample[self.mjj_key]
		#loss = self.loss_strategy(jet_sample)
		train_dataset = tf.data.Dataset.from_tensor_slices((x, loss)).batch(self.batch_sz)

		# build the regressor
		self.regressor = qr.QuantileRegressionV2(**self.model_params)
		self.model = self.regressor.make_model(x_mean_var=(np.mean(x), np.var(x)))
		
		# build the loss and optimizer
		self.loss_function = self.regressor.make_quantile_loss(quantile=self.quantile, y_mean_var=(np.mean(loss), np.var(loss)))
		self.optimizer = tf.keras.optimizers.Adam(0.005)

		# run training
		for epoch in range(self.epochs):
			print("\nStart of epoch %d" % (epoch))
			# Iterate over the batches of the dataset.
			for step, (x_batch, y_batch) in enumerate(train_dataset):
				self.training_step(step, x_batch, y_batch)


	def save(self, path):
		self.model.save(path)

	def load(self, path):
		self.model = tf.keras.models.load_model(path)

	def predict(self, data):
		if isinstance(data, js.JetSample):
			data = data[self.mjj_key]
		predicted = self.model.predict(xx).flatten() 
		return self.unscale_output(predicted)

	def select(self, jet_sample):
		loss_cut = self.predict(jet_sample)
		return self.loss_strategy(jet_sample) > loss_cut

	def __repr__(self):
		return 'QR Cut: ' + Discriminator.__repr__(self)


class GBRDiscriminator(Discriminator):

	def fit(self, jet_sample):
		self.model = scikit.GradientBoostingRegressor(loss='quantile', alpha=1-self.quantile, learning_rate=.01, max_depth=2, verbose=2)
