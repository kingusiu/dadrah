import os

import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import dadrah.util.string_constants_util as stco
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


def save_QR(discriminator, params, experiment, quantile, xsec, model_str=None):
    # save the model   
    model_str = model_str or stco.make_qr_model_str(params.run_n_qr, params.run_n_vae, quantile, params.sig_sample_id, xsec, params.strategy_id)
    model_path = os.path.join(experiment.model_dir_qr, model_str)
    discriminator.save(model_path)
    print('saving model {} to {}'.format(model_str, experiment.model_dir_qr))
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

