import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
import numpy as np
import pathlib
from collections import defaultdict

import pofah.jet_sample as jesa
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.selection.quantile_regression as qure
import dadrah.selection.anomaly_score_strategy as ansc 
import pofah.util.sample_factory as sf
import dadrah.kfold_pipeline.kfold_util as kutil
import dadrah.kfold_pipeline.kfold_string_constants as kstco
import dadrah.selection.losses_and_metrics as lome
import vande.vae.layers as layers



logger = log.get_logger(__name__)

loss_str = 'loss'
metric_str = '2ndDiff'


class SaveBestModelCallback(tf.keras.callbacks.Callback):

    ''' 
        checks
        1. loss <= previous losses, if no: continues, if yes:
        2. metric < previous metrics, if no: continues, if yes:
        saves model
    '''

    def __init__(self,model_dir,*args,**kwargs):
        super(SaveBestModelCallback, self).__init__(*args,**kwargs)
        self.model_dir = model_dir
        self.best_loss_so_far = np.inf
        self.best_metric_so_far = np.inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1:
            self.best_loss_so_far = logs['val_'+loss_str]
            self.best_metric_so_far = logs['val'+metric_str]
        if epoch > 5:
            if logs['val_'+loss_str] <= self.best_loss_so_far and logs['val_'+metric_str] < self.best_metric_so_far:
                self.model.save(model_dir,'best_so_far')


def build_model(loss_fn, metric_fn, layers_n, nodes_n, lr_ini, wd_ini, activation, x_mu_std=(0.0, 1.0), initializer='he_uniform'):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_ini)

    inputs_mjj = tf.keras.Input(shape=(1, ), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1, ), name='targets')
    
    x = inputs_mjj
    x = layers.StdNormalization(*x_mu_std, name='Normalization')(x)
    for i in range(layers_n):
        x = tf.keras.layers.Dense(nodes_n, kernel_initializer=initializer, activation=activation, name='dense_'+str(i+1))(x)

    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, name='dense_out')(x)
    
    model = qure.QrModel(inputs=[inputs_mjj, targets], outputs=outputs)
    model.compile(loss=loss_fn, metric_fn=metric_fn, optimizer=optimizer)
    
    return model


def train_model(params, quantile, qcd_train, qcd_valid, score_strategy, k, tb_dir):


    ### prepare data
    
    x_train, x_valid = qcd_train['mJJ'], qcd_valid['mJJ']
    y_train, y_valid = score_strategy(qcd_train), score_strategy(qcd_valid)
    x_mu_std = (np.mean(x_train), np.std(x_train))

    logger.info('qcd fold {}: min mjj = {}, max mjj = {}'.format(k, np.min(x_train), np.max(x_train)))
    logger.info('qcd fold {}: min loss = {}, max loss = {}'.format(k, np.min(y_train), np.max(y_train)))

    ### build model

    initializer = 'he_uniform'
    regularizer = None
    activation = 'swish'
    wd_ini = 0.0001
    quant_loss = lome.quantile_loss_smooth(quantile) #quantile_loss(params.quantile)
    ratio_metric = lome.scnd_fini_diff_metric() #binned_quantile_dev_loss(params.quantile, accuracy_bins)

    logger.info('loss fun ' + quant_loss.name + ', metric fun ' + ratio_metric.name)

    model = build_model(quant_loss, ratio_metric, layers_n=params.layers_n, nodes_n=params.nodes_n, lr_ini=params.lr, wd_ini=wd_ini, activation=activation, x_mu_std=x_mu_std, initializer=initializer)
    model.summary()

    ### setup callbacks

    tensorboard_callb = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
    es_callb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-08)
 
    ### fit model

    model.fit(x=x_train, y=y_train, batch_size=params.batch_sz, epochs=params.epochs, shuffle=True, validation_data=(x_valid, y_valid), callbacks=[tensorboard_callb, es_callb, reduce_lr], verbose=2)
    
    return model



#**************************************************************************#
#
#
#                               train k QR models
#
#
#**************************************************************************#


def train_k_models(params, qr_model_dir, fig_dir, tb_base_dir, score_strategy_id='rk5_05'):

    #****************************************#
    #           read in all qcd data
    #****************************************#

    qcd_sample_parts = kutil.read_kfold_datasets(kstco.input_dir_kfold, params.kfold_n, read_n=params.read_n)
    score_strategy = ansc.an_score_strategy_dict[score_strategy_id]

    #****************************************#
    #             train and save 5 models
    #****************************************#
        
    model_paths = kutil.get_model_paths(qr_model_dir, params)

    for quantile in params.quantiles:

        q_str = 'q'+str(int(quantile*100))
        logger.info('starting training for quantile ' + q_str)


        for k, qcd_sample_part in zip(range(1,params.kfold_n+1), qcd_sample_parts):

            qcd_train, qcd_valid = jesa.split_jet_sample_train_test(qcd_sample_part, frac=0.7)
            tb_dir = os.path.join(tb_base_dir, q_str,'f'+str(k)) # separate tensorboard dir for every quantile and every fold

            # train qr
            logger.info('training fold no.{} on {} events, validating on {}'.format(k, len(qcd_train), len(qcd_valid)))
            model = train_model(params, quantile, qcd_train, qcd_valid, score_strategy=score_strategy, k=k, tb_dir=tb_dir)

            # save qr
            model_full_path = model_paths[q_str]['fold' + str(k)]
            logger.info('saving model to ' + model_full_path)
            model.save(model_full_path)

            ### write final cut plot

            img_file_writer = tf.summary.create_file_writer(tb_dir)
            img = kutil.plot_discriminator_cut(model, qcd_valid, score_strategy, plot_suffix='_'+q_str+'_fold'+str(k), fig_dir=fig_dir, xlim=False)
            with img_file_writer.as_default():
                tf.summary.image('Test_data_and_cut_' + q_str, img, step=1000) # name of summary groups objects from different directories who share that name together
                img_file_writer.flush()

        # end for each fold of k
    # end for each quantile

    return model_paths

