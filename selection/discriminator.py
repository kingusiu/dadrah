import numpy as np


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
		return 'Flat Cut: '+ str(self.quantile*100) + '% qnt, ' + self.loss_strategy.__name__ + ' strategy'
