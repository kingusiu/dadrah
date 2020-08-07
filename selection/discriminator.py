import numpy as np
import quantile_regression as qr


class Discriminator():

	def __init__(self, quantile, loss_strategy):
		self.loss_strategy = loss_strategy
		self.quantile = quantile
		self.mjj_key = 'mJJ'

	def fit(self, jet_sample):
		pass

	'''
		predict cut for each example in jet_sample
	'''
	def predict(self, jet_sample):
		pass

	def select(self, jet_sample):
		pass

	def __repr__(self):
		pass


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
		return 'Flat Cut: {}% qnt, {} strategy'.format(str(self.quantile*100), self.loss_strategy.__name__)


class QRDiscriminator(Discriminator):

	def __init__(self, *args, **kwargs):
		Discriminator.__init__(*args, **kwargs)
		self.model = qr.QuantileRegression()

	def fit(self, jet_sample):
		loss = self.loss_strategy(jet_sample)
		xx = np.reshape(jet_sample[self.mjj_key], (-1,1))
		self.model.fit(xx, loss, epochs=100, batch_size=128, verbose=2, validation_split=0.2, shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])

	def predict(self, jet_sample):
		xx = np.reshape(jet_sample[self.mjj_key], (-1,1))
		return self.model.predict(xx)

	def select(self, jet_sample):
		