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


class FlatCutDiscriminator(Discriminator):

	def fit(self, jet_sample):
		loss = self.loss_strategy(jet_sample)
		self.cut = np.percentile( loss, (1.-self.quantile)*100 )
		
	def predict(self, jet_sample):
		sz = len(jet_sample[self.mjj_key])
		return np.asarray([self.cut]*sz)

	def select(self, jet_sample):
		loss = self.loss_strategy(jet_sample)
		return loss > self.cut
