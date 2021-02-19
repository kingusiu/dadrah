import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from collections import namedtuple
import datetime
import pathlib
import copy

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import analysis.analysis_discriminator as andi
import vande.training as train
import dadrah.util.data_processing as dapr
import pofah.phase_space.cut_constants as cuts


def make_qr_model_str(run, quantile, sig_id, sig_xsec, date=None):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    return 'QRmodel_run_{}_qnt_{}_{}_sigx_{}_{}.h5'.format(run, str(int(quantile*100)), sig_id, int(sig_xsec), date)


def train_QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=params.strategy, batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = make_qr_model_str(params.run_n, quantile, xsec)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return discriminator


def save_QR(params, experiment, quantile, xsec):
    # save the model   
    model_str = make_qr_model_str(experiment.run_n, quantile, params.sig_sample_id, xsec)
    model_path = os.path.join(experiment.model_dir_qr, model_str)
    discriminator.save(model_path)
    print('saving model {} to {}'.format(model_str, experiment.model_dir_qr))
    return model_path


def load_QR(params, experiment, quantile, xsec, date):
    model_str = make_qr_model_str(experiment.run_n, quantile, params.sig_sample_id, sig_xsec=xsec, date=date)
    model_path = os.path.join(experiment.model_dir_qr, model_str)
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=params.strategy, batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    discriminator.load(model_path)
    return discriminator


def predict_QR(discriminator, sample, inv_quant):
    print('predicting {}'.format(sample.name))
    selection = discriminator.select(sample)
    sample.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)
    return sample


#****************************************#
#           set runtime params
#****************************************#
# for a fixed signal G_RS na 3.5TeV
xsecs = [100., 10., 1., 0.]
# xsecs = [0.]
sig_in_training_nums = [1130, 113, 11, 0]
# sig_in_training_nums = [0]
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
# quantiles = [0.1, 0.99]

Parameters = namedtuple('Parameters','run_n, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy, epochs, read_n')
params = Parameters(run_n=113, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrainReco', 
                    qcd_test_sample_id='qcdSigAllTestReco',
                    sig_sample_id='GtoWW35naReco', 
                    strategy=lost.loss_strategy_dict['rk5'],
                    epochs=100,
                    read_n=None)

make_qcd_train_test_datasample = False
train_qr = True
do_bump_hunt = False
model_path_date = '20210217'

#****************************************#
#           read in data
#****************************************#
experiment = ex.Experiment(params.run_n).setup(model_dir_qr=True, analysis_dir_qr=True)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})

# if datasets not yet prepared, prepare them, dump and return
if make_qcd_train_test_datasample:
    qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, **cuts.signalregion_cuts)
# else read from file
else:
    qcd_train_sample = js.JetSample.from_input_dir(params.qcd_train_sample_id, paths.sample_dir_path(params.qcd_train_sample_id), read_n=params.read_n) 
    qcd_test_sample_ini = js.JetSample.from_input_dir(params.qcd_test_sample_id, paths.sample_dir_path(params.qcd_test_sample_id), read_n=params.read_n)
sig_sample_ini = js.JetSample.from_input_dir(params.sig_sample_id, paths.sample_dir_path(params.sig_sample_id), **cuts.signalregion_cuts)


# ************************************************************
#      run train and/or predict + dijet fit for each xsec
# ************************************************************

for xsec, sig_in_training_num in zip(xsecs, sig_in_training_nums):

    # "reset" samples for new xsec quantile cut results

    model_paths = []
    qcd_test_sample = copy.deepcopy(qcd_test_sample_ini)
    sig_sample = copy.deepcopy(sig_sample_ini)
    if train_qr:
        mixed_train_sample, mixed_valid_sample = dapr.inject_signal(qcd_train_sample, sig_sample_ini, sig_in_training_num)
    
    for quantile in quantiles:

        # using inverted quantile because of dijet fit code
        inv_quant = round((1.-quantile),2)

        # ********************************************
        #               train or load
        # ********************************************

        if train_qr:

            print('training on {} events, validating on {}'.format(len(mixed_train_sample), len(mixed_valid_sample)))

            # train and save QR model
            discriminator = train_QR(quantile, mixed_train_sample, mixed_valid_sample, params)
            discriminator_path = save_QR(params, experiment, quantile, xsec)
            model_paths.append(discriminator_path)

        else: # else load discriminators
            discriminator = load_QR(params, experiment, quantile, xsec, model_path_date)
        
        
        # ********************************************
        #               predict
        # ********************************************
        qcd_test_sample = predict_QR(discriminator, qcd_test_sample, inv_quant)
        sig_sample = predict_QR(discriminator, sig_sample, inv_quant)


    # write results for all quantiles
    result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(params.run_n), '$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(xsec))}) # in selection paths new format with run_x, sig_x, ...
    print('writing selections to ', result_paths.base_dir)
    qcd_test_sample.dump(result_paths.sample_file_path(params.qcd_test_sample_id, mkdir=True))
    sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id))

    # plot results
    discriminator_list = []
    for q, model_path in zip(quantiles, model_paths):
        discriminator = disc.QRDiscriminator_KerasAPI(quantile=q, loss_strategy=params.strategy)
        discriminator.load(model_path)
        discriminator_list.append(discriminator)

    title_suffix = ' trained qcd SR + '+params.sig_sample_id.replace('Reco','')+' at xsec '+str(int(xsec)) + 'fb'
    andi.analyze_multi_quantile_discriminator_cut(discriminator_list, mixed_valid_sample, title_suffix=title_suffix, plot_name='multi_discr_cut_x'+str(int(xsec)), fig_dir=experiment.analysis_dir_qr_cuts)

    # ********************************************
    #               dijet fit
    # ********************************************
    if do_bump_hunt:
        dijet_dir = '/eos/home-k/kiwoznia/dev/vae_dijet_fit/VAEDijetFit'
        cmd = "python run_dijetfit.py --run -n {} -i {} -M 3500 --sig {}.h5 --sigxsec {} --qcd {}.h5".format(params.run_n, result_paths.base_dir, sdfr.path_dict['file_names'][params.sig_sample_id], xsec, sdfr.path_dict['file_names'][params.qcd_test_sample_id])
        print("running ", cmd)
        subprocess.check_call('pwd && source setupenv.sh && ' + cmd, cwd=dijet_dir, shell=True, executable="/bin/bash")  
        print('finished')
