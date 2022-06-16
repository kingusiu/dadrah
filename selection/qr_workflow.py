import os
import json
import numpy as np

import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
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

    discriminator = model_t(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs, n_layers=5, n_nodes=60) if qr_model_t == stco.QR_Model.DENSE else \
        model_t(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs)
    
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
    discriminator.save(model_path)
    print('saving model {} to {}'.format(model_str, model_dir_qr))
    return model_path


def load_QR(params, experiment, quantile, xsec, date, model_str=None):
    model_str = model_str or stco.make_qr_model_str(experiment.run_n, quantile, params.sig_sample_id, sig_xsec=xsec, strategy_id=params.strategy_id, date=date)
    model_path = os.path.join(experiment.model_dir_qr, model_str)
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    discriminator.load(model_path)
    return discriminator


def predict_QR(discriminator, sample, inv_quant):
    print('predicting {}'.format(sample.name))
    selection = discriminator.select(sample)
    sample.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)
    return sample


def fit_polynomial_from_envelope_json(envelope, quantiles, poly_order):

    bin_idx, mu_idx, rmse_idx, min_idx, max_idx = range(5)

    polynomials = {}

    for qq in quantiles:

        qq_key = stco.inv_quantile_str(qq)

        x      = np.array([row[bin_idx] for row in envelope[qq_key]])
        y      = np.array([row[mu_idx] for row in envelope[qq_key]])
        y_down = np.fabs(y-np.array([row[min_idx] for row in envelope[qq_key]]))
        y_up   = np.fabs(y-np.array([row[max_idx] for row in envelope[qq_key]]))

        asymmetric_error = [y_down, y_up]

        coeffs = np.polyfit(x, y, poly_order, w=1/(y_up+y_down))

        polynomials[qq] = np.poly1d(coeffs)

    return polynomials


def fit_polynomial_from_envelope_json(envelope_json_path, quantiles, poly_order):

    ff = open(envelope_json)
    envelope = json.load(ff)

    return fit_polynomial_from_envelope(envelope, quantiles, poly_order)
