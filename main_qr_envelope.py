import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype
import numpy as np
import pathlib
import json

import dadrah.util.data_processing as dapr
import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.util.string_constants_util as stco
import dadrah.selection.qr_workflow as qrwf
import pofah.util.experiment as ex
import analysis.analysis_discriminator as andi


def slice_datasample_n_parts(data, parts_n):
    cuts = np.linspace(0, len(data), num=parts_n+1, endpoint=True, dtype=int)
    return [data.cut(slice(start,stop)) for (start, stop) in zip(cuts[:-1], cuts[1:])]


def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))


# setup runtime params and csv file
quantiles = [0.1, 0.9, 0.99]
parts_n = 5
bin_edges = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                        4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')
bin_centers = [(high+low)/2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]
print('bin centers: ', bin_centers)

Parameters = recordtype('Parameters','run_n, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, strategy_id, epochs, read_n')
params = Parameters(run_n=113, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrainReco', 
                    qcd_test_sample_id='qcdSigAllTestReco',
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=None)

# set directories for saving and loading with extra envelope subdir for qr models
experiment = ex.Experiment(run_n=params.run_n).setup(model_dir_qr=True, analysis_dir_qr=True)
experiment.model_dir_qr = os.path.join(experiment.model_dir_qr, 'envelope')
pathlib.Path(experiment.model_dir_qr).mkdir(parents=True, exist_ok=True)
result_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/run_' + str(params.run_n) + '/envelope'
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)            

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})

data_qcd_all = dapr.merge_qcd_base_and_ext_datasets(params, paths)
print('qcd all: min mjj = {}, max mjj = {}'.format(np.min(data_qcd_all['mJJ']), np.max(data_qcd_all['mJJ'])))
# split qcd data
data_qcd_parts = slice_datasample_n_parts(data_qcd_all, parts_n)

cut_results = {}

#****************************************#
#           for each quantile
#****************************************#
for quantile in quantiles:

    models = []
    cuts = np.empty([0, len(bin_centers)])

    #****************************************#
    #      train, save, predict 5 models
    #****************************************#

    # for each qcd data part
    for dat_train, dat_valid, model_n in zip(data_qcd_parts, data_qcd_parts[1:] + [data_qcd_parts[0]], list('ABCDE')):
        # train qr
        print('training on {} events, validating on {}'.format(len(dat_train), len(dat_valid)))
        discriminator = qrwf.train_QR(quantile, dat_train, dat_valid, params)
        models.append(discriminator)

        # save qr
        model_str = stco.make_qr_model_str(experiment.run_n, quantile, 'no_signal', 0, params.strategy_id)
        model_str = model_str[:-3] + '_' + model_n + model_str[-3:]
        discriminator_path = qrwf.save_QR(discriminator, params, experiment, quantile, 0, model_str)

        # predict cut values per bin
        cuts_part = discriminator.predict(bin_centers)
        cuts = np.append(cuts, cuts_part[np.newaxis,:], axis=0)

    #****************************************#
    #      compute, save and plot envelope
    #****************************************#

    # compute mean, RMS, min, max per bin center over 5 trained models
    mu = np.mean(cuts, axis=0)
    mi = np.min(cuts, axis=0)
    ma = np.max(cuts, axis=0)
    rmse = np.sqrt(np.mean(np.square(cuts-mu), axis=0))
    cuts_for_quantile = np.stack([bin_centers, mu, rmse, mi, ma], axis=1)
    print('cuts for quantile ' + str(quantile) + ': ')
    print(cuts_for_quantile)
    # store cut values in dict
    cut_results.update({inv_quantile_str(quantile): cuts_for_quantile.tolist()})

    # plot quantile cut bands
    title_suffix = ' 5 models trained qcd SR no signal q ' + 'q{:02}'.format(int(quantile*100))
    plot_name = 'multi_discr_cut_no_signal_5models_' + 'q{:02}'.format(int(quantile*100))
    andi.analyze_multi_quantile_discriminator_cut(models, dat_valid, title_suffix=title_suffix, plot_name=plot_name, fig_dir=result_dir)

# write cut result json file
with open(os.path.join(result_dir, 'cut_stats.json'), 'w') as ff:
    json.dump(cut_results, ff)
