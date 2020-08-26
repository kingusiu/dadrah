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

	def __init__(self, *args, n_nodes=20, **kwargs):
		self.n_nodes = n_nodes
		Discriminator.__init__(self, *args, **kwargs)

	def set_mean_var_input_output(self, inp, outp):
		self.mean_inp, self.mean_outp = np.mean(inp), np.mean(outp)
		self.var_inp, self.var_outp = np.var(inp), np.var(outp)

	def scale_input(self, inp):
		inp_scaled = (inp - self.mean_inp) / self.var_inp
		return np.reshape(inp_scaled, (-1,1))

	def scale_output(self, outp):
		return (outp - self.mean_outp) / self.var_outp

	def unscale_output(self, outp):
		return (outp * self.var_outp) + self.mean_outp

	def fit(self, jet_sample):
		self.model = qr.QuantileRegression(quantile=self.quantile, n_nodes=self.n_nodes).build()
		x = jet_sample[self.mjj_key]
		loss = self.loss_strategy(jet_sample)
		self.set_mean_var_input_output(x, loss)
		xx, yy = self.scale_input(x), self.scale_output(loss)
		self.model.fit(xx, yy, epochs=100, batch_size=128, verbose=2, validation_split=0.2, shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])

	def save(self, path):
		self.model.save(path)
	
	def load(self, path):
		self.model = tf.keras.models.load_model(path)

	def predict(self, data):
		if isinstance(data, js.JetSample):
			data = data[self.mjj_key]
		xx = self.scale_input(data)
		predicted = self.model.predict(xx).flatten() 
		return self.unscale_output(predicted)

	def select(self, jet_sample):
		loss_cut = self.predict(jet_sample)
		return self.loss_strategy(jet_sample) > loss_cut

	def __repr__(self):
		return 'QR Cut: ' + Discriminator.__repr__(self)

