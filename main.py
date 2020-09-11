import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.result_writer as reswr
import pofah.util.experiment as ex
import selection.discriminator as dis
import selection.loss_strategy as ls
import analysis.analysis_discriminator as an
import anpofah.util.plotting_util as pu
import pofah.path_constants.sample_dict_file_parts_reco as sd 
import pofah.path_constants.sample_dict_file_parts_selected as sds
import datetime
import dadrah.analysis.root_plotting_util as rpu
import dadrah.selection.selection_util as seu
import dadrah.loss_strategy as lost
from importlib import reload
import os
import setGPU


def make_qr_model_str(train_sz, quantile, date=True):
    date_str = ''
    if date:
        date = datetime.date.today()
        date_str = '_{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    return 'QRmodel_train_sz'+str(int(train_sz*100))+'pc_qnt'+ str(int(quantile*100)) + date_str
    

# read in qcd signal region sample
run_n = 101
SM_sample = 'qcdSigAllReco'
#BSM_samples = ['GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']
BSM_samples = ['GtoWW25naReco', 'GtoWW35naReco']
all_samples = [SM_sample] + BSM_samples
mjj_key = 'mJJ'
reco_loss_j1_key = 'j1RecoLoss'
QR_train_share = 0.3

experiment = ex.Experiment(run_n)
paths = sf.SamplePathDirFactory(sd.path_dict).update_base_path({'$run$': experiment.run_dir})

data = sf.read_inputs_to_jet_sample_dict_from_dir(all_samples, paths)

# define quantile and loss-strategy for discimination
quantiles = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9] # 5%
strategy = lost.loss_strategy_dict['s5'] # L1 & L2 > LT
qcd_sig_sample = data[SM_sample]
#split qcd sample into training and testing
qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sig_sample, QR_train_share)
# update data_dictionary
data[SM_sample] = qcd_test
print(qcd_sig_sample.features())

for quantile in quantiles:

    experiment = ex.Experiment(run_n=run_n, param_dict={{'$quantile$': 'q'+str(int(quantile*100)), '$strategy$': strategy.file_str}}).setup(analysis_dir=True)
    print('writing analysis results to ', experiment.analysis_dir)

    discriminator = dis.QRDiscriminator(quantile=quantile, loss_strategy=strategy, n_nodes=70)
    discriminator.fit(qcd_train)

    # plot mjj qcd sig to check for flat ratio
    an.analyze_discriminator_cut(discriminator, qcd_train, plot_name='discr_cut_qnt'+str(int(quantile*100)), fig_dir=experiment.analysis_dir_fig)

    model_str = make_qr_model_str(QR_train_share, quantile)
    discriminator.save('models/dnn_run_101_{}.h5'.format(model_str))
    print('saving model ', model_str)

    counting_experiment = {}
    bin_edges = [0,1126,1181,1246,1313,1383,1455,1530,1607,1687,1770,1856,1945,2037,2132,2231,2332,2438,2546,2659,2775,2895,3019,3147,3279,3416,3558,3704,3854,4010,4171,4337,4509,4686,4869,5058,5253,5500,5663,5877,6099,6328,6564,6808,1e6]

    # ### qcd training set
    selection = discriminator.select(qcd_train)
    qcd_train.add_feature('sel', selection)
    title = "QCD training set: BG like vs SIG like mjj distribution and their ratio qnt {}".format(int(quantile*100))
    h_bg_like_qcd_train, h_sig_like_qcd_train = rpu.make_bg_vs_sig_ratio_plot(qcd_train.rejected(mjj_key), qcd_train.accepted(mjj_key), target_value=quantile, n_bins=30, title=title, fig_dir=experiment.analysis_dir_fig)

    # ### qcd test set
    sample = data[SM_sample]
    # apply selection to datasample
    selection = discriminator.select(sample)
    sample.add_feature('sel', selection)
    title = "QCD test set: BG like vs SIG like mjj distribution and ratio qnt {}".format(int(quantile*100))
    h_bg_like_qcd_test, h_sig_like_qcd_test = rpu.make_bg_vs_sig_ratio_plot(sample.rejected(mjj_key), sample.accepted(mjj_key), target_value=quantile, n_bins=30, title=title, fig_dir=experiment.analysis_dir_fig)
    # save in counts sig like & bg like for qcd SR test set
    counting_experiment[SM_sample] = seu.get_bin_counts_sig_like_bg_like(sample, bin_edges)

    # apply cuts to signal samples
    for sample_id in BSM_samples:
        # apply selection to datasample
        selection = discriminator.select(data[sample_id])
        data[sample_id].add_feature('sel', selection)

    ### print efficiency table 
    an.print_discriminator_efficiency_table(data)

    ### plot mjj ratio
    for sample_id in BSM_samples:
        sample = data[sample_id]
        title = sample.name + ": BG like vs SIG like mjj distribution and ratio qnt {}".format(int(quantile*100))
        rpu.make_bg_vs_sig_ratio_plot(sample.rejected(mjj_key), sample.accepted(mjj_key), target_value=quantile, n_bins=30, title=title, fig_dir=experiment.analysis_dir_fig)
        # save in counts sig like & bg like for qcd SR test set
        counting_experiment[sample_id] = seu.get_bin_counts_sig_like_bg_like(sample, bin_edges)

    ### write selected samples

    result_paths = sf.SamplePathDirFactory(sds.path_dict).update_base_path({'$run$': experiment.run_dir, '$quantile$': 'q'+str(int(quantile*100)), '$strategy$': strategy.file_str})

    for sample_id, sample in data.items():
        result_file_path = os.path.join(result_paths.sample_dir_path(sample_id), result_paths.sample_file_path(sample_id))
        print('writing results for {} to {}'.format(sds.path_dict['sample_name'][sample_id], result_file_path))
        sample.dump(result_file_path)

    ### write bin counts to file
    reswr.write_bin_counts_to_file(counting_experiment, bin_edges, os.path.join(experiment.analysis_dir_bin_count,'sel_bin_count_'+experiment.run_dir+'_tsz'+str(int(QR_train_share*100))+'pc_q'+str(quantile*100)+'.h5'))
