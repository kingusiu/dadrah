import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
import numpy as np
from collections import defaultdict
import pathlib
import json

import dadrah.selection.quantile_regression as qure
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import vande.vae.layers as layers



logger = log.get_logger(__name__)


#**********************************************************************************#
#
#
#           compute envelope (min,max,mean,rms) on bins from k models
#
#
#**********************************************************************************#

def compute_kfold_envelope(params, model_paths, bin_edges):

    # ********************************************************
    #       load models and compute cuts for bin centers
    # ********************************************************

    cuts_all_models = defaultdict(lambda: np.empty([0, len(bin_edges)]))

    for quantile in params.quantiles:

        q_str = 'q'+str(int(quantile*100))

        for k in range(1,params.kfold_n+1):

            # ***********************
            #       read in model

            model = tf.keras.models.load_model(model_paths[q_str]['fold'+str(k)], custom_objects={'QrModel': qure.QrModel, 'StdNormalization': layers.StdNormalization}, compile=False)

            # predict cut values per bin
            cuts_per_bin = model.predict(bin_edges)
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
            envelopped_cuts = qrwf.calc_cut_envelope(bin_edges, cuts_all_models_fold[mask,...])
            envelope_folds['fold_{}'.format(k+1)][str(quantile)] = envelopped_cuts.tolist()

    # ***********************
    #       save envelope

    envelope_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_'+str(kstco.vae_run_n)+'/qr_run_'+str(params.qr_run_n)+'/sig_'+params.sig_sample_id+'/xsec_'+str(int(params.sig_xsec))+'/loss_rk5_05/envelope_'+str(params.env_n)
    pathlib.Path(envelope_dir).mkdir(parents=True, exist_ok=True)

    for k in range(params.kfold_n+1):
        envelope_json_path = os.path.join(envelope_dir, 'cut_stats_allQ_fold'+str(k+1)+'_'+ params.sig_sample_id + '_xsec_' + str(int(params.sig_xsec)) + '.json')
        logger.info('writing envelope results to ' + envelope_json_path)
        with open(envelope_json_path, 'w') as ff:
            json.dump(envelope_folds['fold_{}'.format(k+1)], ff) # do this separately for each fold (one envelope file per fold)


