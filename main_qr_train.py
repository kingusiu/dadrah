import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from collections import namedtuple

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.loss_strategy as ls
import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost

print(tf.__version__)

#****************************************#
#			set runtime params
#****************************************#
Parameters = namedtuple('Parameters','run_n, sm_sample_id, quantile, strategy, qr_train_share')
params = Parameters(run_n=101, sm_sample_id='qcdSigBisReco', quantile=0.1, strategy=lost.loss_strategy_dict['s5'])

#****************************************#
#			read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
qcd_sig_sample = js.JetSample.from_input_dir(params.sm_sample_id, paths.sample_dir_path(params.sm_sample_id))

#****************************************#
#		train quantile regression
#****************************************#

discriminator = disc.QRDiscriminator(quantile=params.quantile, loss_strategy=params.strategy, epochs=10, n_nodes=40)
discriminator.fit(qcd_sig_sample)
print(discriminator.model.summary())
discriminator.save('./my_new_model.h5')
