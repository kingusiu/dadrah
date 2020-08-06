import numpy as np


class Discriminator():

	def __init__(self, loss_strategy):
		self.loss_strategy = loss_strategy


	def fit(self, jet_sample):
		pass

	def apply(self, jet_sample):
		pass


class FlatCutDiscriminator(Discriminator):

	def fit(self, jet_sample):
		loss = self.loss_strategy(jet_sample)
	