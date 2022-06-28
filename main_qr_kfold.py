import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype
import numpy as np
import pathlib
import json
from collections import defaultdict

import pofah.jet_sample as jesa
import pofah.util.sample_factory as safa
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import pofah.phase_space.cut_constants as cuco
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.util.data_processing as dapr
import dadrah.selection.qr_workflow as qrwf
import dadrah.selection.loss_strategy as lost



def slice_datasample_n_parts(data, parts_n):
    cuts = np.linspace(0, len(data), num=parts_n+1, endpoint=True, dtype=int)
    return [data.filter(slice(start,stop)) for (start, stop) in zip(cuts[:-1], cuts[1:])]


def calc_cut_envelope(cuts):
    # compute mean, RMS, min, max per bin center over all trained models
    mu = np.mean(cuts, axis=0)
    mi = np.min(cuts, axis=0)
    ma = np.max(cuts, axis=0)
    rmse = np.sqrt(np.mean(np.square(cuts-mu), axis=0))
    return np.stack([bin_centers, mu, rmse, mi, ma], axis=1)


#****************************************#
#           set runtime params
#****************************************#

# import ipdb; ipdb.set_trace()
train_discriminator = False
quantiles = [0.3, 0.5, 0.7, 0.9]
resonance = 'na'
sig_xsec = 0
bin_edges = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                        4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')
bin_centers = [(high+low)/2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]
print('bin centers: ', bin_centers)

Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_sample_id, qcd_ext_sample_id, sig_sample_id, strategy_id, epochs, read_n, qr_model_t, poly_order, kfold_n')
params = Parameters(vae_run_n=113,
                    qr_run_n = 31, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    sig_sample_id='GtoWW35'+resonance+'Reco',
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=None,
                    qr_model_t=stco.QR_Model.DENSE, #### **** TODO: update **** ####
                    poly_order=5,
                    kfold_n=5,
                    ) 

loss_function = lost.loss_strategy_dict[params.strategy_id]

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t TRAINING RUN \n'+str(params)+'\n'+'*'*70)

### paths ###

# data inputs: /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$ 
# data outputs (selections): /eos/user/k/kiwoznia/data/QR_results/events/vae_run_$run_n_vae$/qr_run_$run_n_qr$/sig_GtoWW35naReco/xsec_100/loss_rk5_05
# model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$

qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)
pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)
envelope_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)+'/sig_GtoWW35naReco/xsec_'+str(sig_xsec)+'/loss_rk5_05/envelope'
pathlib.Path(envelope_dir).mkdir(parents=True, exist_ok=True)
polys_json_path = os.path.join(envelope_dir, 'polynomials_allQ_allFolds_'+ params.sig_sample_id + '_xsec_' + str(sig_xsec) + '.json')
param_dict = {'$vae_run_n$': str(params.vae_run_n), '$qr_run_n$': str(params.qr_run_n), '$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(sig_xsec)), '$loss_strat$': params.strategy_id}
result_paths = safa.SamplePathDirFactory(sdfs.path_dict).update_base_path(param_dict) # in selection paths new format with run_x, sig_x, ...

#****************************************#
#           read in all qcd data
#****************************************#
input_paths = safa.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.vae_run_n)})

qcd_sample_all = dapr.merge_qcd_base_and_ext_datasets(params, input_paths, **cuco.signalregion_cuts)
logger.info('qcd all: min mjj = {}, max mjj = {}'.format(np.min(qcd_sample_all['mJJ']), np.max(qcd_sample_all['mJJ'])))
logger.info('qcd all: min loss = {}, max loss = {}'.format(np.min(loss_function(qcd_sample_all)), np.max(loss_function(qcd_sample_all))))
# split qcd data
qcd_sample_parts = slice_datasample_n_parts(qcd_sample_all, params.kfold_n)

#****************************************#
#             train 5 models
#****************************************#

if train_discriminator:

    cuts_all_models = defaultdict(lambda: np.empty([0, len(bin_centers)]))

    for quantile in quantiles:

        for k, qcd_sample_part in zip(range(1,params.kfold_n+1), qcd_sample_parts):
        
            qcd_train, qcd_valid = jesa.split_jet_sample_train_test(qcd_sample_part, frac=0.7)

            # train qr
            logger.info('training fold no.{} on {} events, validating on {}'.format(k, len(qcd_train), len(qcd_valid)))
            discriminator = qrwf.train_QR(quantile, qcd_train, qcd_valid, params, qr_model_t=params.qr_model_t)

            # save qr
            model_str = stco.make_qr_model_str(run_n_qr=params.qr_run_n, run_n_vae=params.vae_run_n, quantile=quantile, sig_id=params.sig_sample_id, sig_xsec=sig_xsec, strategy_id=params.strategy_id)
            model_str = model_str[:-3] + '_fold' + str(k) + model_str[-3:]
            discriminator_path = qrwf.save_QR(discriminator, params, qr_model_dir, quantile, sig_xsec, model_str)

            # predict cut values per bin
            cuts_per_bin = discriminator.predict(bin_centers)
            cuts_all_models[quantile] = np.append(cuts_all_models[quantile], cuts_per_bin[np.newaxis,:], axis=0)

        # end for each fold of k
    # end for each quantile

    #***********************************************************#
    #         compute envelope: mean, min, max, rms cuts
    #***********************************************************#

    envelope_folds = defaultdict(dict)

    # compute average cut for each fold 

    for k in range(params.kfold_n+1):

        mask = np.ones(params.kfold_n, dtype=bool)
        if k < params.kfold_n: mask[k] = False

        for quantile in quantiles:

            cuts_all_models_fold = cuts_all_models[quantile]
            envelopped_cuts = calc_cut_envelope(cuts_all_models_fold[mask,...])
            envelope_folds['fold_{}'.format(k+1)][quantile] = envelopped_cuts.tolist()

    # write out envelope for each fold
    for k in range(params.kfold_n+1):
        envelope_json_path = os.path.join(envelope_dir, 'cut_stats_allQ_fold'+str(k+1)+'_'+ params.sig_sample_id + '_xsec_' + str(sig_xsec) + '.json')
        logger.info('writing envelope results to ' + envelope_json_path)
        with open(envelope_json_path, 'w') as ff:
            json.dump(envelope_folds['fold_{}'.format(k+1)], ff) # do this separately for each fold (one envelope file per fold)


    #************************************************************#
    #               fit polynomials from envelope
    #************************************************************#

    polynomials_folds = {} # per kfold -> per quantile
    for k in range(1,params.kfold_n+2):

        polynomials = qrwf.fit_polynomial_from_envelope(envelope_folds['fold_{}'.format(k)], quantiles, params.poly_order)
        polynomials_folds['fold_{}'.format(k)] = polynomials

    # write out polynomials to json file in same dir as envelope (1 file for all folds)
    logger.info('writing polynomials to ' + polys_json_path)
    # import ipdb; ipdb.set_trace()
    dapr.write_polynomials_to_json(polys_json_path, polynomials_folds)

else: # else read polynomials from file

    logger.info('reading polynomials from ' + polys_json_path)
    polynomials_folds = dapr.read_polynomials_from_json(polys_json_path, quantiles, params.kfold_n)


#************************************************************#
#      predict: make selections from fitted polynomials
#************************************************************#

logger.info('applying discriminator cuts for selection')

sig_sample = jesa.JetSample.from_input_dir(params.sig_sample_id, input_paths.sample_dir_path(params.sig_sample_id), read_n=params.read_n, **cuco.signalregion_cuts)

for k, sample in zip(range(1, params.kfold_n+2), qcd_sample_parts+[sig_sample]):
    for quantile in quantiles:
        selection = qrwf.fitted_selection(sample, params.strategy_id, quantile, polynomials_folds['fold_{}'.format(k)])
        sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)


#************************************************************#
#                   write selection results
#************************************************************#

qcd_sample_results = qcd_sample_parts[0]
for k in range(1, params.kfold_n):
    qcd_sample_results = qcd_sample_results.merge(qcd_sample_parts[k])

qcd_sample_results.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True))

sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id))
