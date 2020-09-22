import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from collections import namedtuple

import dadrah.selection.loss_strategy as lost
import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import dadrah.selection.discriminator as disc
import pofah.path_constants.sample_dict_file_parts_reco as sdfr


#****************************************#
#			set runtime params
#****************************************#
Parameters = namedtuple('Parameters','run_n, sample_id, quantile, strategy')
params = Parameters(run_n=101, sample_id='qcdSigBisReco', quantile=0.1, strategy=lost.loss_strategy_dict['s5'])

#****************************************#
#			read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
sample = js.JetSample.from_input_dir(params.sample_id, paths.sample_dir_path(params.sample_id))

#****************************************#
#		load quantile regression
#****************************************#
discriminator = disc.QRDiscriminator(quantile=params.quantile, loss_strategy=params.strategy)
discriminator.load('./my_new_model.h5')

#****************************************#
#		apply quantile regression
#****************************************#
selection = discriminator.select(sample)
sample.add_feature('sel', selection)
sample.dump('./qcd_sig_bis_sel.h5')

