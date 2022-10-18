import os
import json
import numpy as np

import dadrah.selection.discriminator as disc
import dadrah.selection.anomaly_score_strategy as ansc
import dadrah.util.string_constants as stco
import vande.training as train


discriminator_dict = {
    stco.QR_Model.DENSE : disc.QRDiscriminator_KerasAPI,    
    stco.QR_Model.POLY : disc.QRDiscriminatorPoly_KerasAPI,
    stco.QR_Model.BERNSTEIN : disc.QRDiscriminatorBernstein_KerasAPI
}


def train_QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False, qr_model_t=stco.QR_Model.POLY):

    # train QR on qcd-signal-injected sample and quantile q

    model_t = discriminator_dict[qr_model_t]

    discriminator = model_t(quantile=quantile, loss_strategy=ansc.an_score_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs, n_layers=5, n_nodes=60) if qr_model_t == stco.QR_Model.DENSE else \
        model_t(quantile=quantile, loss_strategy=ansc.an_score_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs)
    
    print('\ntraining {} QR for quantile {}'.format(type(discriminator), quantile))    
    
    losses_train, losses_valid = discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return discriminator


def save_QR(discriminator, params, model_dir_qr, quantile, xsec, model_str=None):
    # save the model   
    model_str = model_str or stco.make_qr_model_str(params.run_n_qr, params.run_n_vae, quantile, params.sig_sample_id, xsec, params.strategy_id)
    model_path = os.path.join(model_dir_qr, model_str)
    print('saving model {} to {}'.format(model_str, model_dir_qr))
    discriminator.save(model_path, include_optimizer=False) # TODO: problems saving adamw
    return model_path


def load_QR(params, model_dir_qr, quantile, xsec, model_str=None):
    model_str = model_str or stco.make_qr_model_str(params.run_n_qr, params.run_n_vae, quantile, params.sig_sample_id, sig_xsec=xsec, strategy_id=params.strategy_id)
    model_path = os.path.join(model_dir_qr, model_str)
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=ansc.an_score_strategy_dict[params.strategy_id], batch_sz=256)
    discriminator.load(model_path)
    return discriminator


def predict_QR(discriminator, sample, quantile):
    print('predicting {}'.format(sample.name))
    selection = discriminator.select(sample)
    sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)
    return sample


def calc_cut_envelope(bin_centers, cuts):
    # compute mean, RMS, min, max per bin center over all trained models
    mu = np.mean(cuts, axis=0)
    mi = np.min(cuts, axis=0)
    ma = np.max(cuts, axis=0)
    rmse = np.sqrt(np.mean(np.square(cuts-mu), axis=0))
    return np.stack([bin_centers, mu, rmse, mi, ma], axis=1)



def fit_polynomial_from_envelope(envelope, quantiles, poly_order): # -> dict(np.poly1d)

    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)

    polynomials = {}

    for qq in quantiles:

        qq_key = str(qq)
        env_qq = np.asarray(envelope[qq_key])

        x      = env_qq[:,bin_idx]
        y      = env_qq[:,mu_idx]
        y_down = np.fabs(y-env_qq[:,min_idx])
        y_up   = np.fabs(y-env_qq[:,max_idx])

        asymmetric_error = [y_down, y_up]

        coeffs = np.polyfit(x, y, poly_order, w=1/(y_up+y_down))

        polynomials[qq] = np.poly1d(coeffs)

    return polynomials


def fit_polynomial_from_envelope_json(envelope_json, quantiles, poly_order):

    ff = open(envelope_json)
    envelope = json.load(ff)

    return fit_polynomial_from_envelope(envelope, quantiles, poly_order)


def fitted_selection(sample, strategy_id, quantile, polynomials, xshift=0.):
    loss_strategy = ansc.an_score_strategy_dict[strategy_id]
    loss = loss_strategy(sample)
    loss_cut = polynomials[quantile]
    return loss > loss_cut(sample['mJJ']-xshift) # shift x by min mjj for bias fixed lmfits


def ensemble_selection(sample, strategy_id, quantile, model_ensemble, fold_i, kfold_n, norm_x, norm_y):

    voting_n = kfold_n-1 if fold_i < kfold_n else kfold_n # signal fold = dummy fold no kfold_n+1 -> all kfold_n models voting
    mjj = norm_x.transform(sample['mJJ'].reshape(-1,1)).squeeze()
    score_strategy = ansc.an_score_strategy_dict[strategy_id]
    an_score = score_strategy(sample)
    an_score = norm_y.transform(an_score.reshape(-1,1)).squeeze()

    an_score_cut = np.zeros(len(sample))
    for j in range(1, kfold_n+1):
        if j != fold_i:
            an_score_cut += model_ensemble['fold_{}'.format(j)].predict([mjj,an_score]).flatten()
    an_score_cut /= voting_n

    return an_score > an_score_cut

