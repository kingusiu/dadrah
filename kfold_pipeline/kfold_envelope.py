import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
import numpy as np
from collections import defaultdict
import pathlib
import json
import matplotlib.pyplot as plt

import dadrah.selection.quantile_regression as qure
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.kfold_pipeline.kfold_string_constants as kstco
import dadrah.selection.qr_workflow as qrwf
import vande.vae.layers as layers



logger = log.get_logger(__name__)


def calc_cut_envelope(bin_centers, cuts):
    # compute mean, RMS, min, max per bin center over all trained models
    mu = np.mean(cuts, axis=0)
    mi = np.min(cuts, axis=0)
    ma = np.max(cuts, axis=0)
    rmse = np.sqrt(np.mean(np.square(cuts-mu), axis=0))
    return np.stack([bin_centers, mu, rmse, mi, ma], axis=1)


# *********************************************************************************#
#                               plotting


def calc_relative_uncertainties(envelope_per_fold, quantiles, params):
    
    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)
    rel_uncert_per_fold = {}
    
    for k in range(params.kfold_n+1):
        
        envelope_fold = envelope_per_fold['fold_{}'.format(k+1)]
        rel_uncert_per_quant = {}
    
        for q in quantiles:
            
            envelope_q = np.asarray(envelope_fold[str(q)])
            
            uncert = (envelope_q[:,max_idx]-envelope_q[:,min_idx])/envelope_q[:,mu_idx]
                        
            rel_uncert_per_quant[str(q)] = uncert
            
        rel_uncert_per_fold['fold_{}'.format(k+1)] = rel_uncert_per_quant
            
    return rel_uncert_per_fold



def calc_uncertainty_band_per_quantile(uncert_per_fold, quantiles, params, bins):
    
    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)
    
    uncert_band_per_quantile = {}
    
    for q in quantiles:
        
        uu = np.empty([0, len(bins)])
        
        for k in range(params.kfold_n):
        
            uncert_fold = uncert_per_fold['fold_{}'.format(k+1)]     
            uncert_q = np.asarray(uncert_fold[str(q)])
            
            uu = np.append(uu, uncert_q[np.newaxis,:], axis=0)
            
        min_all_folds = np.min(uu, axis=0).tolist()
        max_all_folds = np.max(uu, axis=0).tolist()
        mu_all_folds = np.mean(uu, axis=0).tolist()
        
        uncert_band_per_quantile[str(q)] = (min_all_folds, max_all_folds, mu_all_folds)
        
    return uncert_band_per_quantile



def plot_uncertainty_band_per_quantile(uncert_band_per_quantile, quantiles, params, bins, fig_dir):
    
    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)
    
    fig, axs = plt.subplots(1, len(quantiles), figsize=(20,4), sharex=True, sharey=True)
    
    for q, ax in zip(quantiles, axs.flat):
        
        mins, maxs, mus = uncert_band_per_quantile[str(q)]
            
        ax.plot(bins, mus, lw=1.5)
        ax.fill_between(bins, mins, maxs, alpha=0.4, linewidth=0)
        ax.set_yscale('log')
            
        ax.set_title('Q {}'.format(q))
        ax.grid()
        ax.set_xlabel('mJJ')
        axs.flat[0].set_ylabel('min \& max around mu')
        #ax.set_xlim(right=4000)
        #ax.set_ylim(top=3)
        
    plt.savefig(fig_dir+'uncertainty_band_quantiles.pdf')



def plot_envelope_uncerts(envelope_per_fold, quantiles, params, fig_dir):
    
    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)
    
    fig, axs = plt.subplots(1, params.kfold_n+1, figsize=(30,4), sharex=True, sharey=True)
    
    for k, ax in zip(range(params.kfold_n+1), axs.flat):
        
        envelope_fold = envelope_per_fold['fold_{}'.format(k+1)]
    
        for q in quantiles:
            
            envelope_q = np.asarray(envelope_fold[str(q)])
            
            x = envelope_q[:,bin_idx]
            y = (envelope_q[:,max_idx]-envelope_q[:,min_idx])/envelope_q[:,mu_idx]
            
            ax.plot(x, y, lw=1.5, label='Q '+str(q))
            ax.set_yscale('log')
            
        ax.set_title('fold {}'.format(k+1))
        ax.legend()
        ax.grid()
        ax.set_xlabel('mJJ')
        axs.flat[0].set_ylabel(r'$\frac{\textrm{max}-\textrm{min}}{\mu}$')
        #ax.set_xlim(right=5000)
        #ax.set_ylim(top=0.1)
        
    plt.savefig(fig_dir+'relative_uncertainty_quantiles.pdf')



#**********************************************************************************#
#
#
#           compute envelope (min,max,mean,rms) on bins from k models
#
#
#**********************************************************************************#

def compute_kfold_envelope(params, model_paths, bin_edges):

    # figure path
    fig_dir = '../fig/env_analysis/env_'+str(params.env_n) +'/'

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
            cuts_per_bin = np.squeeze(model.predict([bin_edges,bin_edges]))
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

        for quantile in params.quantiles:

            cuts_all_models_fold = cuts_all_models[quantile]
            envelopped_cuts = calc_cut_envelope(bin_edges, cuts_all_models_fold[mask,...])
            envelope_folds['fold_{}'.format(k+1)][str(quantile)] = envelopped_cuts.tolist()

    
    # ***********************
    #       plot results

    plot_envelope_uncerts(envelope_folds, params.quantiles, params, fig_dir)

    rel_uncert_per_fold = calc_relative_uncertainties(envelope_folds, quantiles, params)
    uncert_band_per_quantile = calc_uncertainty_band_per_quantile(rel_uncert_per_fold, quantiles, params, bin_edges)
    plot_uncertainty_band_per_quantile(uncert_band_per_quantile, quantiles, params, bin_edges)


    # ***********************
    #       save envelope

    envelope_dir = kutil.get_envelope_dir(params)

    for k in range(params.kfold_n+1):
        envelope_json_path = os.path.join(envelope_dir, 'cut_stats_allQ_fold'+str(k+1)+'_'+ params.sig_sample_id + '_xsec_' + str(int(params.sig_xsec)) + '.json')
        logger.info('writing envelope results to ' + envelope_json_path)
        with open(envelope_json_path, 'w') as ff:
            json.dump(envelope_folds['fold_{}'.format(k+1)], ff) # do this separately for each fold (one envelope file per fold)


    return envelope_dir