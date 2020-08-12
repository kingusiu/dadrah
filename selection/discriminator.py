import numpy as np
import tensorflow as tf
import dadrah.selection.quantile_regression as qr
import pofah.jet_sample as js


class Discriminator():

	def __init__(self, quantile, loss_strategy):
		self.loss_strategy = loss_strategy
		self.quantile = quantile
		self.mjj_key = 'mJJ'

	def fit(self, jet_sample):
		pass

	def save(self, path):
		pass

	def load(self, path):
		pass 

	'''
		predict cut for each example in data
		data ... jet_sample instance or raw numpy array
	'''
	def predict(self, data):
		pass

	def select(self, jet_sample):
		pass

	def __repr__(self):
		return '{}% qnt, {} strategy'.format(str(self.quantile*100), self.loss_strategy.__name__)


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

	def __init__(self, *args, **kwargs):
		Discriminator.__init__(self, *args, **kwargs)

	def fit(self, jet_sample):
		self.model = qr.QuantileRegression(self.quantile).build()
		loss = self.loss_strategy(jet_sample)
		xx = np.reshape(jet_sample[self.mjj_key], (-1,1))
		self.model.fit(xx, loss, epochs=10, batch_size=128, verbose=2, validation_split=0.2, shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])

	def save(self, path):
		self.model.save(path)
	
	def load(self, path):
		self.model = tf.keras.models.load_model(path)

	def predict(self, data):
		if isinstance(data, js.JetSample):
			data = data[self.mjj_key]
		xx = np.reshape(data, (-1,1))
		return self.model.predict(xx).flatten()


	def select(self, jet_sample):
		loss_cut = self.predict(jet_sample)
		return self.loss_strategy(jet_sample) > loss_cut

	def __repr__(self):
		return 'QR Cut: ' + Discriminator.__repr__(self)

