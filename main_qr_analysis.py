import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import namedtuple

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import dadrah.selection.loss_strategy as lost
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.discriminator as disc
import analysis.analysis_discriminator as andi


#****************************************#
#			set runtime params
#****************************************#
Parameters = namedtuple('Parameters','run_n, sm_sample_id, quantile, strategy')
params = Parameters(run_n=101, sm_sample_id='qcdSigBisReco', quantile=0.1, strategy=lost.loss_strategy_dict['s5'])

#****************************************#
#			read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
qcd_sig_sample = js.JetSample.from_input_dir(params.sm_sample_id, paths.sample_dir_path(params.sm_sample_id))

#****************************************#
#		load quantile regression
#****************************************#
discriminator = disc.QRDiscriminator(quantile=params.quantile, loss_strategy=params.strategy)
discriminator.load('./my_new_model.h5')

#****************************************#
#		load quantile regression
#****************************************#
andi.analyze_discriminator_cut(discriminator, qcd_sig_sample, plot_name='discr_cut_qnt'+str(int(params.quantile*100)), fig_dir='.')
