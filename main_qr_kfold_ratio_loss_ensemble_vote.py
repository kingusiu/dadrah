import os
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
import tensorflow_addons as tfa
from recordtype import recordtype
import numpy as np
import pathlib
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pofah.jet_sample as jesa
import pofah.util.sample_factory as safa
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import pofah.phase_space.cut_constants as cuco
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.util.data_processing as dapr
import dadrah.selection.quantile_regression as qure
import dadrah.selection.qr_workflow as qrwf
import dadrah.selection.anomaly_score_strategy as ansc
import dadrah.analysis.analysis_discriminator as andi
import pofah.util.sample_factory as sf


def read_kfold_datasets(path, kfold_n, read_n): # -> list(jesa.JetSample)
    file_names = ['qcd_sqrtshatTeV_13TeV_PU40_NEW_fold'+str(k+1)+'.h5' for k in range(kfold_n)]
    print('reading ' + ' '.join(file_names) + 'from ' + path)
    return [jesa.JetSample.from_input_file('qcd_fold'+str(k+1),os.path.join(path,ff), read_n=read_n) for k,ff in enumerate(file_names)]


def make_model(layers_n, nodes_n, initializer, activation): # -> tf.keras.Model

    inputs_mjj = tf.keras.Input(shape=(1,), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1,), name='targets') # only needed for calculating metric because update() signature is limited to y & y_pred in keras.metrics class 
    x = inputs_mjj

    for i in range(layers_n):
        x = tf.keras.layers.Dense(nodes_n, kernel_initializer=initializer, activation=activation, name='dense'+str(i+1))(x)

    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, name='dense'+str(layers_n+1))(x)

    model = qure.QrModel(name='QR', inputs=[inputs_mjj,targets], outputs=outputs)

    return model


def plot_losses(history, plot_name_suffix, fig_dir):

    for loss_type in ['total_loss', 'quant_loss', 'ratio_loss']:

        loss_train = history.history[loss_type]
        loss_valid = history.history['val_'+loss_type]

        fig = plt.figure()
        plt.semilogy(loss_train)
        plt.semilogy(loss_valid)
        plt.title(loss_type.replace('_',' '))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training','validation'], loc='upper right')
        plt.savefig(os.path.join(fig_dir,loss_type+'_'+plot_name_suffix+'.png'))
        plt.close()


#****************************************#
#           set runtime params
#****************************************#

# import ipdb; ipdb.set_trace()
# quantiles = [0.3, 0.5, 0.7, 0.9]
quantiles = [0.3, 0.9]
resonance = 'na'
# sig_sample_ids = ['GtoWW15'+resonance+'Reco', 'GtoWW25'+resonance+'Reco', 'GtoWW35'+resonance+'Reco', 'GtoWW45'+resonance+'Reco']
sig_sample_ids = ['GtoWW35'+resonance+'Reco']
sig_xsec = 0

Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_sample_id, qcd_ext_sample_id, sig_sample_id, strategy_id, read_n, kfold_n')
params = Parameters(vae_run_n=113,
                    qr_run_n = 192, ####!!! **** TODO: update **** !!!####
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    read_n=int(1e4),
                    kfold_n=5,
                    ) 

score_strategy = ansc.an_score_strategy_dict[params.strategy_id]
accuracy_bins = np.array([1199., 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928]).astype('float')


#****************************************#
#           model parameters
#****************************************#
HyperParams = recordtype('HyperParams','layers_n nodes_n batch_sz initializer lr wd activation epochs_n')
hyperp = HyperParams(layers_n=4, nodes_n=8, batch_sz=32, initializer='he_uniform', lr=1e-3, wd=1e-5, activation='swish', epochs_n=2)
optimizer = tfa.optimizers.AdamW(learning_rate=hyperp.lr, weight_decay=hyperp.wd)

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t TRAINING RUN \n'+str(params)+'\n'+'*'*70)
logger.info('\n'+'*'*70+'\n'+'\t\t\t hyper-parameters \n'+str(hyperp)+'\n'+'*'*70)

### paths ###

# data inputs: /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$/5folds for qcd and /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$/ for signal
# data outputs (selections): /eos/user/k/kiwoznia/data/QR_results/events/vae_run_$run_n_vae$/qr_run_$run_n_qr$/sig_GtoWW35naReco/xsec_100/loss_rk5_05
# model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$

input_paths_sig = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.vae_run_n)})

qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)
pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)
fig_dir = 'fig'+'/qr_run_'+str(params.qr_run_n)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
envelope_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)+'/sig_GtoWW35naReco/xsec_'+str(sig_xsec)+'/loss_rk5_05/envelope'
pathlib.Path(envelope_dir).mkdir(parents=True, exist_ok=True)
polys_json_path = os.path.join(envelope_dir, 'polynomials_allQ_allFolds_'+ params.sig_sample_id + '_xsec_' + str(sig_xsec) + '.json')


# dummy selection result paths since no signal injected yet
param_dict = {'$vae_run_n$': str(params.vae_run_n), '$qr_run_n$': str(params.qr_run_n), '$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(sig_xsec)), '$loss_strat$': params.strategy_id}
result_paths = safa.SamplePathDirFactory(sdfs.path_dict).update_base_path(param_dict) # in selection paths new format with run_x, sig_x, ...


#****************************************#
#           read in all qcd data
#****************************************#

input_dir_kfold = '/eos/user/k/kiwoznia/data/VAE_results/events/run_113/qcd_sqrtshatTeV_13TeV_PU40_NEW_'+str(params.kfold_n)+'fold_signalregion_parts'

qcd_sample_parts = read_kfold_datasets(input_dir_kfold, params.kfold_n, read_n=params.read_n)

#****************************************#
#           setup normalization
#****************************************#

mjj_full = qcd_sample_parts[0]['mJJ']
score_full = score_strategy(qcd_sample_parts[0])
for k in range(1, params.kfold_n):
    mjj_full = np.append(mjj_full, qcd_sample_parts[k]['mJJ'])
    score_full = np.append(score_full, score_strategy(qcd_sample_parts[k]))

norm_x_mima = preprocessing.MinMaxScaler()
norm_y_mima = preprocessing.MinMaxScaler()

norm_x_mima.fit(mjj_full.reshape(-1,1))
norm_y_mima.fit(score_full.reshape(-1,1))

# normalize bins
accuracy_bins = norm_x_mima.transform(accuracy_bins.reshape(-1,1)).squeeze()


#****************************************#
#             train 5 models
#****************************************#

models = defaultdict(dict) # model 2-level-dict: [quantile_i][fold_j]

for quantile in quantiles:

    # instantiate loss functions
    quant_loss = qure.quantile_loss(quantile)
    ratio_loss = qure.quantile_dev_loss(quantile) # qure.binned_quantile_dev_loss(quantile, accuracy_bins) #  # #None

    for k, qcd_sample_part in zip(range(1,params.kfold_n+1), qcd_sample_parts):

        logger.info('qcd fold {}: min mjj = {}, max mjj = {}'.format(k, np.min(qcd_sample_part['mJJ']), np.max(qcd_sample_part['mJJ'])))
        logger.info('qcd fold {}: min loss = {}, max loss = {}'.format(k, np.min(score_strategy(qcd_sample_part)), np.max(score_strategy(qcd_sample_part))))

        qcd_train, qcd_valid = jesa.split_jet_sample_train_test(qcd_sample_part, frac=0.7)

        #****************************************#
        #           prepare inputs & targets
        #****************************************#
        score_strategy = ansc.an_score_strategy_dict[params.strategy_id]
        x_train, x_valid = qcd_train['mJJ'], qcd_valid['mJJ']
        y_train, y_valid = score_strategy(qcd_train), score_strategy(qcd_valid) 


        x_train_mima, x_valid_mima = norm_x_mima.transform(x_train.reshape(-1,1)).squeeze(), norm_x_mima.transform(x_valid.reshape(-1,1)).squeeze()
        y_train_mima, y_valid_mima = norm_y_mima.transform(y_train.reshape(-1,1)).squeeze(), norm_y_mima.transform(y_valid.reshape(-1,1)).squeeze()

        # train qr
        logger.info('training fold no.{} on {} events, validating on {}'.format(k, len(qcd_train), len(qcd_valid)))
        qr_model = make_model(hyperp.layers_n, hyperp.nodes_n, hyperp.initializer, hyperp.activation)
        qr_model.summary()

        es_callb = tf.keras.callbacks.EarlyStopping(monitor="val_total_loss", patience=7)

        qr_model.compile(optimizer=optimizer, quant_loss=quant_loss, ratio_loss=ratio_loss, run_eagerly=True)
        history = qr_model.fit(x=x_train_mima, y=y_train_mima, batch_size=hyperp.batch_sz, epochs=hyperp.epochs_n, 
            validation_data=(x_valid_mima, y_valid_mima), callbacks=[es_callb])

        # plot training loss and discriminator cut
        plot_losses(history, 'q'+str(int(quantile*100))+'_fold'+str(k), fig_dir)

        # save qr
        models[str(quantile)]['fold_{}'.format(k)] = qr_model
        model_str = stco.make_qr_model_str(run_n_qr=params.qr_run_n, run_n_vae=params.vae_run_n, quantile=quantile, sig_id=params.sig_sample_id, sig_xsec=sig_xsec, strategy_id=params.strategy_id)
        model_str = model_str[:-3] + '_fold' + str(k) + model_str[-3:]
        discriminator_path = qrwf.save_QR(qr_model, params, qr_model_dir, quantile, sig_xsec, model_str)

    # end for each fold of k
# end for each quantile

#************************************************************#
#                       ensemble predict qcd
#************************************************************#

# import ipdb; ipdb.set_trace()
logger.info('applying QR to qcd sample')

# predict ad write qcd
for k, sample in zip(range(1, params.kfold_n+1), qcd_sample_parts):
    for quantile in quantiles:
        model_ensemble = models[str(quantile)]
        import ipdb; ipdb.set_trace()
        selection = qrwf.ensemble_selection(sample, params.strategy_id, quantile, model_ensemble, k, params.kfold_n, norm_x_mima, norm_y_mima)
        sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)

qcd_sample_results = qcd_sample_parts[0]
for k in range(1, params.kfold_n):
    qcd_sample_results = qcd_sample_results.merge(qcd_sample_parts[k])

qcd_sample_results.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True))

#************************************************************#
#                   ensemble predict signal
#************************************************************#
# predict and write signals

sig_kfold_n = params.kfold_n+1

for sig_sample_id in sig_sample_ids:

    logger.info('applying QR to ' + str(sig_sample_id))

    sig_sample = jesa.JetSample.from_input_dir(sig_sample_id, input_paths_sig.sample_dir_path(sig_sample_id), read_n=params.read_n, **cuco.signalregion_cuts)

    for quantile in quantiles:
        model_ensemble = models[str(quantile)]
        selection = qrwf.ensemble_selection(sig_sample, params.strategy_id, quantile, model_ensemble, sig_kfold_n, params.kfold_n, norm_x_mima, norm_y_mima)
        sig_sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)

    sig_sample.dump(result_paths.sample_file_path(sig_sample_id, mkdir=True))


#************************************************************#
#                   plot ensemble cuts
#************************************************************#

logger.info('plotting ensemble cuts')

andi.analyze_multi_quantile_ensemble_cut(models, qcd_sample_results, quantiles, params.kfold_n, params.strategy_id, 
    norm_x_mima, norm_y_mima, plot_name='ensemble_multi_quantile_cut', fig_dir=fig_dir, cut_xmax=False, cut_ymax=False)
