import os
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype
import numpy as np
import pathlib
import json
from collections import defaultdict

import pofah.jet_sample as jesa
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.selection.quantile_regression as qure
import dadrah.selection.anomaly_score_strategy as lost
import pofah.util.sample_factory as sf
import dadrah.kfold_pipeline.kfold_util as kutil
import dadrah.kfold_pipeline.kfold_string_constants as kstco
import dadrah.selection.losses_and_metrics as lome


logger = log.get_logger(__name__)




def build_model(quantile_loss, ratio_metric, layers_n, nodes_n, lr_ini, wd_ini, activation, x_mu_std=(0.0, 1.0), initializer='glorot_uniform'):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_ini)

    inputs_mjj = tf.keras.Input(shape=(1, ), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1, ), name='targets')
    
    x = inputs_mjj
    x = layers.StdNormalization(*x_mu_std, name='Normalization')(x)
    for _ in range(layers_n):
        x = tf.keras.layers.Dense(nodes_n, kernel_initializer=initializer, activation=activation)(x)

    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
    
    model = qure.QrModel(inputs=[inputs_mjj, targets], outputs=outputs)
    model.compile(loss=quantile_loss, ratio_metric=ratio_metric, optimizer=optimizer)
    
    return model


def train_model(params, quantile, qcd_train, qcd_valid, qcd_test_sample):

    #****************************************#
    #           build model
    #****************************************#

    layers_n = 4
    nodes_n = 8
    initializer = 'he_uniform'
    regularizer = None
    activation = 'swish'
    lr_ini = 0.001
    wd_ini = 0.0001
    quant_loss = lome.quantile_loss_smooth(quantile) #quantile_loss(params.quantile)
    ratio_metric = lome.scnd_fini_diff_metric() #binned_quantile_dev_loss(params.quantile, accuracy_bins)

    logger.info('loss fun ' + quant_loss.name + ', metric fun ' + ratio_metric.name)

    model = build_model(quant_loss, ratio_metric, layers_n, nodes_n, lr_ini=lr_ini, wd_ini=wd_ini, activation=activation, x_mu_std=x_mu_std, initializer=initializer)
    model.summary()

    ### setup callbacks

    tensorboard_callb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    es_callb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-08)
    img_log_dir = tensorboard_log_dir + '/plots'
    os.system('rm -rf ' + img_log_dir + '/*')
    plot_cb = PlotCutCallback(img_log_dir, qcd_test_sample, score_strategy)
    
    ### fit model

    model.fit(x=x_train, y=y_train, batch_size=params.batch_sz, epochs=params.epochs, shuffle=True, validation_data=(x_test, y_test),callbacks=[tensorboard_callb, es_callb, reduce_lr, plot_cb])
    
    return model



#**************************************************************************#
#
#
#                               train k QR models
#
#
#**************************************************************************#


def train_k_models(params):

    #****************************************#
    #           read in all qcd data
    #****************************************#

    qcd_sample_parts = kutil.read_kfold_datasets(kstco.input_dir_kfold, params.kfold_n)

    #****************************************#
    #             train and save 5 models
    #****************************************#
        

    cuts_all_models = defaultdict(lambda: np.empty([0, len(bin_centers)]))

    for quantile in params.quantiles:

        for k, qcd_sample_part in zip(range(1,params.kfold_n+1), qcd_sample_parts):

            logger.info('qcd fold {}: min mjj = {}, max mjj = {}'.format(k, np.min(qcd_sample_part['mJJ']), np.max(qcd_sample_part['mJJ'])))
            logger.info('qcd fold {}: min loss = {}, max loss = {}'.format(k, np.min(loss_function(qcd_sample_part)), np.max(loss_function(qcd_sample_part))))

            qcd_train, qcd_valid = jesa.split_jet_sample_train_test(qcd_sample_part, frac=0.7)

            # train qr
            logger.info('training fold no.{} on {} events, validating on {}'.format(k, len(qcd_train), len(qcd_valid)))
            model = train_model(quantile, qcd_train, qcd_valid, params, qr_model_t=params.qr_model_t)

            # save qr
            model_str = stco.make_qr_model_str(run_n_qr=params.qr_run_n, run_n_vae=kstco.vae_run_n, quantile=quantile, sig_id=params.sig_sample_id, sig_xsec=sig_xsec, strategy_id=params.strategy_id)
            model_str = model_str[:-3] + '_fold' + str(k) + model_str[-3:]
            model_path = qrwf.save_QR(model, params, qr_model_dir, quantile, sig_xsec, model_str)

            ### save model
    
            model_str = stco.make_qr_model_str(run_n_qr=(params.qr_run_n), run_n_vae=(params.vae_run_n), quantile=(params.quantile), sig_id=(params.sig_sample_id), sig_xsec=0, strategy_id=(params.strategy_id))
            logger.info('saving model to ' + model_str)
            model.save(os.path.join(qr_model_dir, model_str))

            ### write final cut plot

            img_file_writer = tf.summary.create_file_writer(img_log_dir)
            img = plot_discriminator_cut(model, qcd_test_sample, score_strategy, fig_dir=fig_dir,xlim=False)
            with img_file_writer.as_default():
                tf.summary.image('Training data and cut', img, step=1000)

            
        # end for each fold of k
    # end for each quantile


#**********************************************************************************#
#
#
#           compute envelope (min,max,mean,rms) on bins from k models
#
#
#**********************************************************************************#

def compute_kfold_envelope(params, bin_edges, quantiles):

    # ********************************************************
    #       load models and compute cuts for bin centers
    # ********************************************************

    cuts_all_models = defaultdict(lambda: np.empty([0, len(bin_edges)]))

    for quantile in quantiles:

        for k in range(1,params.kfold_n+1):

            # ***********************
            #       read in models

            model_str = stco.make_qr_model_str(run_n_qr=params.in_qr_run_n, run_n_vae=params.vae_run_n, quantile=quantile, sig_id=params.sig_sample_id, sig_xsec=params.sig_xsec, strategy_id=params.strategy_id)
            model_str = model_str[:-3] + '_fold' + str(k) + model_str[-3:]

            discriminator = qrwf.load_QR(params, in_qr_model_dir, quantile, params.sig_xsec, model_str=model_str)

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

    envelope_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.envelope_out_qr_run_n)+'/sig_'+params.sig_sample_id+'/xsec_'+str(int(params.sig_xsec))+'/loss_rk5_05/envelope'
    pathlib.Path(envelope_dir).mkdir(parents=True, exist_ok=True)

    for k in range(params.kfold_n+1):
        envelope_json_path = os.path.join(envelope_dir, 'cut_stats_allQ_fold'+str(k+1)+'_'+ params.sig_sample_id + '_xsec_' + str(int(params.sig_xsec)) + '.json')
        logger.info('writing envelope results to ' + envelope_json_path)
        with open(envelope_json_path, 'w') as ff:
            json.dump(envelope_folds['fold_{}'.format(k+1)], ff) # do this separately for each fold (one envelope file per fold)







#****************************************#
#           set runtime params
#****************************************#

# import ipdb; ipdb.set_trace()
train_discriminator = False
quantiles = [0.3, 0.5, 0.7, 0.9]
resonance = 'br'
sig_sample_ids = ['GtoWW15'+resonance+'Reco', 'GtoWW25'+resonance+'Reco', 'GtoWW35'+resonance+'Reco', 'GtoWW45'+resonance+'Reco']
sig_xsec = 0
bin_edges = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                        4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')
bin_centers = [(high+low)/2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]
print('bin centers: ', bin_centers)

Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_sample_id, qcd_ext_sample_id, sig_sample_id, strategy_id, epochs, read_n, qr_model_t, poly_order, kfold_n, xshift')
params = Parameters(vae_run_n=113,
                    qr_run_n = 34, ####!!! **** TODO: update **** !!!####
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=None,
                    qr_model_t=stco.QR_Model.DENSE,
                    poly_order=5,
                    kfold_n=5,
                    xshift=1227.5, ####!!! **** TODO: update **** !!!#### 1200 or 1227.5 for all lmfit polynomials (fixed bias), 0 otherwise
                    ) 

loss_function = lost.an_score_strategy_dict[params.strategy_id]

# logging
logger.info('\n'+'*'*70+'\n'+'\t\t\t TRAINING RUN \n'+str(params)+'\n'+'*'*70)

### paths ###

# data inputs: /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$/5folds for qcd and /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$/ for signal
# data outputs (selections): /eos/user/k/kiwoznia/data/QR_results/events/vae_run_$run_n_vae$/qr_run_$run_n_qr$/sig_GtoWW35naReco/xsec_100/loss_rk5_05
# model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$

input_paths_sig = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.vae_run_n)})

qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)
pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)
envelope_dir = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)+'/sig_GtoWW35naReco/xsec_'+str(sig_xsec)+'/loss_rk5_05/envelope'
pathlib.Path(envelope_dir).mkdir(parents=True, exist_ok=True)
polys_json_path = os.path.join(envelope_dir, 'polynomials_allQ_allFolds_'+ params.sig_sample_id + '_xsec_' + str(sig_xsec) + '.json')



