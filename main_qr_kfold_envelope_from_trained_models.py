import os
import numpy as np
from recordtype import recordtype
import json
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from enum import Enum
from collections import defaultdict

import dadrah.selection.qr_workflow as qrwf
import dadrah.util.string_constants as stco
import dadrah.selection.qr_workflow as qrwf
import dadrah.selection.loss_strategy as lost
import dadrah.util.logging as log

eps = 1e-6

def fit_poly_from_envelope(fit_fun, uncert_fun, degree, envelope, quantiles, *fit_args):
    
    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)

    fits = {}

    for qq in quantiles:

        qq_key = str(qq)
        env_qq = np.asarray(envelope[qq_key])

        x      = env_qq[:,bin_idx]
        y      = env_qq[:,mu_idx]
        uncert = uncert_fun(env_qq)        
        cc = fit_fun(degree, x, y, uncert, *fit_args)
        
        fits[qq] = np.poly1d(cc)

    return fits


def fit_poly_from_envelope_forall_folds(fit_fun, uncert_fun, degree, envelope_per_fold, quantiles, params, *fit_args):
    
    poly_fits_per_fold = {}
    
    for k in range(params.kfold_n+1):
        poly_fits = fit_poly_from_envelope(fit_fun, uncert_fun, degree, envelope_per_fold['fold_{}'.format(k+1)], quantiles, *fit_args)
        poly_fits_per_fold['fold_{}'.format(k+1)] = poly_fits
    
    return poly_fits_per_fold


def residual(params, x, y, uncert, x_shift):
    y_hat = np.poly1d(params)(x-x_shift) # shift first bin to zero & fix bias
    return (y - y_hat)/uncert


def fit_lm(degree, x, y, uncert, x_shift):
    
    params_lmfit = Parameters()
    for d in range(degree-1):
        params_lmfit.add('c'+str(d), value=1)
    # add fixed bias value
    params_lmfit.add('bias', value=y[0], vary=False)
    
    return minimize(residual, params_lmfit, args=(x, y, uncert, x_shift))


def fit_lm_coeff(degree, x, y, uncert, x_shift):
    
    out = fit_lm(degree, x, y, uncert, x_shift)
    
    return list(out.params.valuesdict().values())


def uncertainty_minmax(envelope, fix_point_n=0):
    sigma = envelope[:,max_idx]-envelope[:,min_idx]
    sigma[:fix_point_n] = eps
    return sigma


def plot_poly_fits(envelope_per_fold, poly_fits_per_fold, quantiles, params, plot_name_suffix, x_shift=0):
    
    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)
    
    for q in quantiles:
        
        fig, axs = plt.subplots(2, params.kfold_n+1, figsize=(30,6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, sharey='row')
        
        for k, ax, ax_ratio in zip(range(params.kfold_n+1), axs.flat, axs.flat[int(len(axs.flat)/2):]):
            
            envelope_q = np.asarray(envelope_per_fold['fold_{}'.format(k+1)][str(q)])
            poly_fit_q = poly_fits_per_fold['fold_{}'.format(k+1)][q]
            
            x = envelope_q[:,bin_idx]
            y = envelope_q[:,mu_idx]
            y_hat = poly_fit_q(x-x_shift)
            yerr = [y-envelope_q[:,min_idx], envelope_q[:,max_idx]-y]
            
            ax.errorbar(x, y, yerr=yerr, fmt='o', ms=1.5, zorder=1)
            ax.plot(x, y_hat, c='r',lw=1, zorder=2)
            ax.set_title('fold {}'.format(k+1))
            
            ax_ratio.plot(x, (y-y_hat)/uncertainty_minmax(envelope_q), 'o', ms=1.7)
            ax_ratio.grid(True, which='major', axis='y')
            #ax_ratio.set_ylim([0.995,1.005])
        
        for ax in axs.flat:
            ax.label_outer()
        
        plt.suptitle('quantile {}'.format(q))
        plt.savefig(fig_dir+'poly_fit_q{}_{}.pdf'.format(int(q*100), plot_name_suffix))


def compute_lm_fits(degree, envelope_per_fold, quantiles, params, x_shift):
    fit_fun = fit_lm_coeff
    uncert_fun = uncertainty_minmax #uncertainty_stddev #uncertainty_yerr #uncertainty_rmse
    return fit_poly_from_envelope_forall_folds(fit_fun, uncert_fun, degree, envelope_per_fold, quantiles, params, x_shift)


# ***********************
#       parameters
# ***********************

binning_t = Enum('binning_t', 'LINEAR DIJET EXPO')

sig_xsec = 0.
quantiles = [0.3, 0.5, 0.7, 0.9]

Parameters = recordtype('Parameters','vae_run_n, in_qr_run_n, envelope_out_qr_run_n, poly_out_qr_run_n, sig_sample_id, strategy_id, kfold_n, bins_n, mjj_min, mjj_max, envelope_binning, degree')
params = Parameters(vae_run_n=113,
                    in_qr_run_n = 32, #### **** TODO: update **** ####
                    envelope_out_qr_run_n = 47, #### **** TODO: update **** #### envelope number from model number 'in_qr_run_n'
                    poly_out_qr_run_n = 48, #### **** TODO: update **** #### polyfit number from envelope number envelope_out_qr_run_n from model number 'in_qr_run_n'
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    kfold_n=5,
                    bins_n=40,
                    mjj_min=1200.,
                    mjj_max=6900.,
                    envelope_binning=binning_t.DIJET,
                    degree=11, # polynomial degree
                    ) 

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t envelope calculation \n'+str(params)+'\n'+'*'*70)

# ***********************
#       paths
# ***********************

in_qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.in_qr_run_n)
out_qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.envelope_out_qr_run_n)
fig_dir = 'fig/poly_analysis/'
pathlib.Path(out_qr_model_dir).mkdir(parents=True, exist_ok=True)


# ***********************
#       define bins
# ***********************

if params.envelope_binning == binning_t.DIJET:
    bin_edges = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                        4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')
    bin_edges = bin_edges[bin_edges>=params.mjj_min]

elif params.envelope_binning == binning_t.EXPO:
    expo_x_shift = 3
    lin_bins = np.linspace(0.,1.,params.bins_n)
    exp_bins = lin_bins/(np.exp(-lin_bins+expo_x_shift)/np.exp(expo_x_shift-1))
    bin_edges = exp_bins*(params.mjj_max-params.mjj_min)+params.mjj_min
    
else: # simple linear binning
    bin_edges = np.array(np.linspace(params.mjj_min, params.mjj_max, params.bins_n).tolist()).astype('int') #100 GeV binning. Stop at 5600! Fit fails if going to 6800

bin_centers = [(high+low)/2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]
logger.info('bin edges: ', bin_edges)


# ********************************************************
#       load models and compute cuts for bin centers
# ********************************************************

cuts_all_models = defaultdict(lambda: np.empty([0, len(bin_edges)]))

for quantile in quantiles:

        for k in range(1,params.kfold_n+1):

            # ***********************
            #       read in models

            model_str = stco.make_qr_model_str(run_n_qr=params.in_qr_run_n, run_n_vae=params.vae_run_n, quantile=quantile, sig_id=params.sig_sample_id, sig_xsec=sig_xsec, strategy_id=params.strategy_id)
            model_str = model_str[:-3] + '_fold' + str(k) + model_str[-3:]

            discriminator = qrwf.load_QR(params, in_qr_model_dir, quantile, sig_xsec, model_str=model_str)

            # predict cut values per bin
            cuts_per_bin = discriminator.predict(bin_edges)
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

envelope_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.envelope_out_qr_run_n)+'/sig_GtoWW35naReco/xsec_'+str(int(sig_xsec))+'/loss_rk5_05/envelope'
pathlib.Path(envelope_dir).mkdir(parents=True, exist_ok=True)

for k in range(params.kfold_n+1):
    envelope_json_path = os.path.join(envelope_dir, 'cut_stats_allQ_fold'+str(k+1)+'_'+ params.sig_sample_id + '_xsec_' + str(int(sig_xsec)) + '.json')
    logger.info('writing envelope results to ' + envelope_json_path)
    with open(envelope_json_path, 'w') as ff:
        json.dump(envelope_folds['fold_{}'.format(k+1)], ff) # do this separately for each fold (one envelope file per fold)



# ***********************
#       fit polynomials
# ***********************

polys_json_path = os.path.join(envelope_dir, 'polynomials_allQ_allFolds_'+ params.sig_sample_id + '_xsec_' + str(params.sig_xsec) + '.json')

x_shift = np.asarray(envelope_folds['fold_1'][str(quantiles[0])])[0,bin_idx]
bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)

lm_fits_per_fold = compute_lm_fits(params.degree, envelope_per_fold, quantiles, params, x_shift)
plot_poly_fits(envelope_per_fold, lm_fits_per_fold, quantiles, params, 'lmfit_ord'+str(params.degree)+'_qr'+str(params.poly_out_qr_run_n), x_shift)

# write polynomials to file
dapr.write_polynomials_to_json(make_polys_json_path(params.poly_out_qr_run_n), lm_fits_per_fold)

