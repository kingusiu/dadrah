import os
import setGPU

from collections import namedtuple

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import dadrah.selection.loss_strategy as lost
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.discriminator as disc
import analysis.analysis_discriminator as andi


single_discriminator_analysis = False
multi_discriminator_analysis = True


#****************************************#
#			set runtime params
#****************************************#
Parameters = namedtuple('Parameters','run_n, sm_sample_id, quantile, strategy')
params = Parameters(run_n=101, sm_sample_id='qcdSigAllReco', quantile=0.1, strategy=lost.loss_strategy_dict['rk5'])

#****************************************#
#			read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
qcd_sig_sample = js.JetSample.from_input_dir(params.sm_sample_id, paths.sample_dir_path(params.sm_sample_id))


if single_discriminator_analysis:

	#****************************************#
	#		load quantile regression
	#****************************************#
	discriminator = disc.QRDiscriminator(quantile=params.quantile, loss_strategy=params.strategy)
	discriminator.load('./my_new_model.h5')

	#****************************************#
	#		load quantile regression
	#****************************************#
	andi.analyze_discriminator_cut(discriminator, qcd_sig_sample, plot_name='discr_cut_qnt'+str(int(params.quantile*100)), fig_dir='.')


if multi_discriminator_analysis:

	discriminator_list = []

	quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
	models = ['models/dnn_run_101_QRmodel_train_sz30pc_qnt'+str(q)+'_20200912.h5' for q in [10, 30, 50, 70, 90]]

	for q, model_path in zip(quantiles,models):
		discriminator = disc.QRDiscriminator(quantile=q, loss_strategy=params.strategy)
		discriminator.load(model_path)
		discriminator.set_mean_var_input_output(qcd_sig_sample['mJJ'], discriminator.loss_strategy(qcd_sig_sample))
		discriminator_list.append(discriminator)

	andi.analyze_multi_quantile_discriminator_cut(discriminator_list, qcd_sig_sample, fig_dir='.')
