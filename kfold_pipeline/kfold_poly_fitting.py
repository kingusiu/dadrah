import os


eps = 1e-6


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


def compute_lm_fits(degree, envelope_per_fold, quantiles, params, x_shift):
    fit_fun = fit_lm_coeff
    uncert_fun = uncertainty_minmax #uncertainty_stddev #uncertainty_yerr #uncertainty_rmse
    return fit_poly_from_envelope_forall_folds(fit_fun, uncert_fun, degree, envelope_per_fold, quantiles, params, x_shift)



#**********************************************************************************#
#
#
#               fit polynomials for all quantiles from envelope
#
#
#**********************************************************************************#


def fit_polynomials(paramsm, envelope_dir):

    # ***********************
    #       read envelope
    # ***********************

    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)

    envelope_per_fold = {}
    for k in range(params.kfold_n+1):
        envelope_json_path = os.path.join(envelope_dir, 'cut_stats_allQ_fold'+str(k+1)+'_'+ params.sig_sample_id + '_xsec_' + str(params.sig_xsec) + '.json')
        ff = open(envelope_json_path)
        envelope_per_fold['fold_{}'.format(k+1)] = json.load(ff)
        
    x_shift = np.asarray(envelope_per_fold['fold_1'][str(quantiles[0])])[0,bin_idx]

    lm_fits_per_fold = compute_lm_fits(params.degree, envelope_per_fold, quantiles, params, x_shift)
    plot_poly_fits(envelope_per_fold, lm_fits_per_fold, quantiles, params, 'lmfit_ord'+str(params.degree)+'_qr'+str(params.poly_out_qr_run_n), x_shift)

    # write polynomials to file
    dapr.write_polynomials_to_json(make_polys_json_path(params.poly_out_qr_run_n), lm_fits_per_fold)
