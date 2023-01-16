import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters

import dadrah.util.data_processing as dapr
import dadrah.kfold_pipeline.kfold_util as kutil
import dadrah.kfold_pipeline.kfold_string_constants as kstco


eps = 1e-6

bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)


# ******************************************
#               uncertainties

def uncertainty_minmax(envelope):
    return envelope[:,max_idx]-envelope[:,min_idx]


def uncertainty_yerr(envelope):
    y = envelope[:,mu_idx]
    return np.asarray([y-envelope[:,min_idx], envelope[:,max_idx]-y])


def uncertainty_updown(envelope):
    y      = envelope[:,mu_idx]
    y_down = np.fabs(y-envelope[:,min_idx])
    y_up   = np.fabs(y-envelope[:,max_idx])
    return y_down+y_up


# ******************************************
#               lmfit


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



def fit_poly_from_envelope(fit_fun, uncert_fun, degree, envelope, quantiles, *fit_args):

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


def compute_lm_fits(degree, envelope_per_fold, quantiles, params, x_shift):
    fit_fun = fit_lm_coeff
    uncert_fun = uncertainty_yerr #uncertainty_minmax #uncertainty_stddev #uncertainty_yerr #uncertainty_rmse
    return fit_poly_from_envelope_forall_folds(fit_fun, uncert_fun, degree, envelope_per_fold, quantiles, params, x_shift)


# *********************************************************************************#
#                               plotting

def plot_poly_fits(envelope_per_fold, poly_fits_per_fold, quantiles, params, plot_name_suffix, x_shift=0, fig_dir='fig'):
    
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
        plt.savefig(os.path.join(fig_dir,'poly_fit_q{}_{}.pdf'.format(int(q*100), plot_name_suffix)))




#**********************************************************************************#
#
#
#               fit polynomials for all quantiles from envelope
#
#
#**********************************************************************************#


def fit_kfold_polynomials(params, envelope_dir):

    # paths: polynomial jsons and figures
    fig_dir = kstco.get_polynomials_fig_dir(params)

    # ***********************
    #       read envelope
    # ***********************

    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)

    envelope_per_fold = {}
    for k in range(1,params.kfold_n+2): # k: 1-6
        envelope_json_path = kstco.get_envelope_file(params,k)
        ff = open(envelope_json_path)
        envelope_per_fold['fold_{}'.format(k)] = json.load(ff)
        
    x_shift = np.asarray(envelope_per_fold['fold_1'][str(params.quantiles[0])])[0,bin_idx]

    lm_fits_per_fold = compute_lm_fits(params.poly_order, envelope_per_fold, params.quantiles, params, x_shift)
    plot_poly_fits(envelope_per_fold, lm_fits_per_fold, params.quantiles, params, 'lmfit_ord'+str(params.poly_order)+'_poly'+str(params.poly_run_n), x_shift, fig_dir)

    # write polynomials to file
    polys_json_path = kstco.get_polynomials_full_file_path(params)
    dapr.write_polynomials_to_json(polys_json_path, lm_fits_per_fold, x_shift)

    return polys_json_path
