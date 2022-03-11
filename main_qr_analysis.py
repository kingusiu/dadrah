import os
import setGPU

from collections import namedtuple

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import dadrah.selection.loss_strategy as lost
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.discriminator as disc
import dadrah.util.string_constants_util as stco
import analysis.analysis_discriminator as andi


single_discriminator_analysis = False
multi_discriminator_analysis = True

#****************************************#
#			set runtime params
#****************************************#
strategy_id = 'rk5_05'
Parameters = namedtuple('Parameters','run_n, sm_sample_id, quantile, strategy, sig_sample_id, sig_xsec')
params = Parameters(run_n=113, sm_sample_id='qcdSigExtReco', quantile=0.9, strategy=lost.loss_strategy_dict[strategy_id], \
					sig_sample_id='GtoWW35naReco', sig_xsec=100)

#****************************************#
#			read in data
#****************************************#

param_dict = {'$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(params.sig_xsec)), '$loss_strat$': strategy_id}
experiment = ex.Experiment(run_n=params.run_n, param_dict=param_dict).setup(model_dir_qr=True, analysis_dir_qr=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
qcd_sig_sample = js.JetSample.from_input_dir(params.sm_sample_id, paths.sample_dir_path(params.sm_sample_id))


if single_discriminator_analysis:

	#****************************************#
	#		load quantile regression
	#****************************************#
	discriminator = disc.QRDiscriminator(quantile=params.quantile, loss_strategy=params.strategy)
	discriminator.load('./my_new_model.h5')

	#****************************************#
	#		analyze quantile regression
	#****************************************#
	andi.analyze_discriminator_cut(discriminator, qcd_sig_sample, plot_name='discr_cut_qnt'+str(int(params.quantile*100)), fig_dir='.')


if multi_discriminator_analysis:

	discriminator_list = []

	quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
	date = '20220303'
	models = [os.path.join(experiment.model_dir_qr, stco.make_qr_model_str(params.run_n, q, params.sig_sample_id, params.sig_xsec, strategy_id, date)) \
				for q in quantiles]

	for q, model_path in zip(quantiles,models):
		discriminator = disc.QRDiscriminator_KerasAPI(quantile=q, loss_strategy=params.strategy, batch_sz=256, n_layers=5, n_nodes=60)
		discriminator.load(model_path)
		discriminator_list.append(discriminator)

	title_suffix = 'QR trained on QCD + '+params.sig_sample_id.replace('Reco','')+' at xsec '+str(int(params.sig_xsec)) + 'fb'
	plot_name = 'multi_discr_cut_'+params.sig_sample_id.replace('Reco','')+'_x'+str(int(params.sig_xsec))+'_loss_'+strategy_id+'_fullX'

	andi.analyze_multi_quantile_discriminator_cut(discriminator_list, qcd_sig_sample, title_suffix=title_suffix, \
												plot_name=plot_name, fig_dir=experiment.analysis_dir_qr_cuts, cut_xmax=False)
