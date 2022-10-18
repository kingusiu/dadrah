# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.6 (default, Nov 29 2020, 15:03:39) 
# [Clang 12.0.0 (clang-1200.0.32.27)]
# Embedded file name: /eos/home-k/kiwoznia/dev/data_driven_anomaly_hunting/dadrah/playground/qr_playground_model.py
# Compiled at: 2022-10-05 15:41:59
# Size of source mod 2**32: 17192 bytes
import os, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU, tensorflow as tf, tensorflow_addons as tfa, kerastuner as kt
from recordtype import recordtype
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from sklearn import preprocessing
import pathlib, dadrah.playground.playground_util as pgut, dadrah.selection.anomaly_score_strategy as ansc, dadrah.util.logging as log, dadrah.util.string_constants as stco, vande.vae.layers as layers

class quantile_loss(tf.keras.losses.Loss):

    def __init__(self, quantile, name='quantileLoss'):
        super().__init__(name=name)
        self.quantile = tf.constant(quantile)

    @tf.function
    def call(self, targets, predictions):
        targets = tf.squeeze(targets)
        predictions = tf.squeeze(predictions)
        err = tf.subtract(targets, predictions)
        return tf.reduce_mean(tf.where(err >= 0, self.quantile * err, (self.quantile - 1) * err))


class quantile_dev_loss:

    def __init__(self, quantile, name='quantileDevLoss'):
        self.name = name
        self.quantile = tf.constant(quantile)

    @tf.function
    def __call__(self, inputs, targets, predictions):
        predictions = tf.squeeze(predictions)
        count_tot = tf.shape(inputs)[0]
        count_above = tf.math.count_nonzero(tf.math.greater(targets, predictions))
        ratio = tf.math.divide_no_nan(tf.cast(count_above, tf.float32), tf.cast(count_tot, tf.float32))
        return tf.math.square(self.quantile - ratio)


class binned_quantile_dev_loss:

    def __init__(self, quantile, bins, name='binnedQuantileDevLoss'):
        self.name = name
        self.quantile = tf.constant(quantile)
        self.bins_n = len(bins)
        self.bins = tf.constant(bins.astype('float32'))

    def __call__(self, inputs, targets, predictions):
        predictions = tf.squeeze(predictions)
        bin_idcs = tf.searchsorted(self.bins, inputs)
        ratios = tf.Variable([self.quantile] * self.bins_n)
        for bin_idx in range(1, self.bins_n + 1):
            bin_mask = tf.math.equal(bin_idcs, bin_idx)
            count_tot = tf.math.count_nonzero(bin_mask)
            if count_tot > 0:
                count_bel = tf.math.count_nonzero(targets[bin_mask] < predictions[bin_mask])
                ratio = tf.math.divide_no_nan(tf.cast(count_bel, tf.float32), tf.cast(count_tot, tf.float32))
                ratios[(bin_idx - 1)].assign(ratio)

        return tf.reduce_sum(tf.math.square(ratios - self.quantile))


class scnd_fini_diff_metric():

    def __init__(self, delta=1e-2, name='smooth'):
        self.name=name
        self.delta = tf.constant(delta) # delta to approximate second derivative

    # @tf.function
    def __call__(self, pred, pred_delta_plus, pred_delta_minus): # for integration in regular TF -> compute predictions for delta-shifted inputs in outside train/test step
        # import ipdb; ipdb.set_trace()
        pred = tf.squeeze(pred)
        pred_delta_plus = tf.squeeze(pred_delta_plus) # targets input not used in prediction
        pred_delta_minus = tf.squeeze(pred_delta_minus)
        
        # 2nd finite diff
        fini_diff2 = tf.math.divide_no_nan((pred_delta_plus - tf.cast(tf.constant(2.0),tf.float32)*pred + pred_delta_minus),tf.math.square(self.delta))  

        return tf.reduce_mean(tf.math.square(fini_diff2)) # mean per batch


class QrModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        (super().__init__)(*args, **kwargs)

    def compile(self, loss, ratio_metric, optimizer, run_eagerly=True, **kwargs):
        (super().compile)(optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)
        self.quant_loss_fn = loss
        self.ratio_metric_fn = ratio_metric
        self.loss_mean = tf.keras.metrics.Mean('loss')
        self.ratio_metric_mean = tf.keras.metrics.Mean(self.ratio_metric_fn.name)

    def train_step(self, data):

        inputs, targets = data
        
        with tf.GradientTape() as (tape):
            predictions = self([inputs, targets], training=True)
            loss = self.quant_loss_fn(targets, predictions)
        
        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        if self.ratio_metric_fn.name == 'smooth':
            delta = self.ratio_metric_fn.delta
            pred_delta_plus = self([inputs+delta, targets], training=False)
            pred_delta_minus = self([inputs-delta, targets], training=False)
            metric_val = self.ratio_metric_fn(predictions, pred_delta_plus, predictions_delta_minus)

        else:
            inputs_norm = self.get_layer('Normalization')(inputs)
            metric_val = self.ratio_metric_fn(inputs_norm, targets, predictions)

        self.loss_mean.update_state(loss)
        self.ratio_metric_mean.update_state(metric_val)

        return {'loss':self.loss_mean.result(), self.ratio_metric_fn.name : self.ratio_metric_mean.result()}


    def test_step(self, data):

        inputs, targets = data

        predictions = self([inputs, targets], training=False)

        loss = self.quant_loss_fn(targets, predictions)

        if self.ratio_metric_fn.name == 'smooth':
            delta = self.ratio_metric_fn.delta
            pred_delta_plus = self([inputs+delta, targets], training=False)
            pred_delta_minus = self([inputs-delta, targets], training=False)
            import ipdb; ipdb.set_trace()
            metric_val = self.ratio_metric_fn(predictions, pred_delta_plus, predictions_delta_minus)

        else:
            inputs_norm = self.get_layer('Normalization')(inputs)
            metric_val = self.ratio_metric_fn(inputs_norm, targets, predictions)

        self.loss_mean.update_state(loss)
        self.ratio_metric_mean.update_state(metric_val)
        
        return {'loss':self.loss_mean.result(), self.ratio_metric_fn.name : self.ratio_metric_mean.result()}

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_mean)
        metrics.append(self.ratio_metric_mean)
        return metrics


class LogTransform(tf.keras.layers.Layer):

    def __init__(self, x_min, **kwargs):
        (super(LogTransform, self).__init__)(**kwargs)
        self.x_min = x_min

    def get_config(self):
        config = super(LogTransform, self).get_config()
        config.update({'x_min': self.x_min})
        return config

    def call(self, x):
        return tf.math.log(x - self.x_min + 1.0)


def build_model(quantile_loss, ratio_metric, layers_n, nodes_n, lr_ini, wd_ini, activation, x_mu_std=(0.0, 1.0), x_min=0.0, norm='std', initializer='glorot_uniform'):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_ini)
    inputs_mjj = tf.keras.Input(shape=(1, ), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1, ), name='targets')
    x = inputs_mjj
    if norm == 'std':
        x = (layers.StdNormalization)(*x_mu_std, **{'name': 'Normalization'})(x)
    else:
        x = LogTransform(x_min, name='Normalization')(x)
    for _ in range(layers_n):
        x = tf.keras.layers.Dense(nodes_n, kernel_initializer=initializer, activation=activation)(x)

    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
    model = QrModel(inputs=[inputs_mjj, targets], outputs=outputs)
    model.compile(loss=quantile_loss, ratio_metric=ratio_metric, optimizer=optimizer)
    return model


def plot_log_transformed_results(model, x_train, y_train, fig_dir):

    x_transformed = model.get_layer('Normalization')(x_train).numpy()
    
    plt.hist(x_transformed, bins=100)
    plt.savefig(fig_dir + '/mjj_transformed.png')
    plt.close()
    
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(x_transformed)
    x_max = np.max(x_transformed)
    plt.hist2d(x_transformed, y_train, range=((x_min * 0.9, np.percentile(x_transformed, 99.999)), (np.min(y_train), np.percentile(y_train, 99.99))),
      norm=(LogNorm()),bins=100)
    xs = np.arange(np.min(x_train), np.max(x_train), 0.001 * (np.max(x_train) - np.min(x_train)))
    xs_transformed = model.get_layer('Normalization')(xs).numpy()
    plt.plot(xs_transformed, (model.predict([xs, xs])), '-', color='m', lw=2.5, label='selection cut')
    plt.colorbar()
    plt.savefig(fig_dir + '/mjj_transformed_score_plane_with_cut.png')
    plt.close()


def plot_discriminator_cut(discriminator, sample, score_strategy, feature_key='mJJ', plot_name='discr_cut', fig_dir=None, plot_suffix=''):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])
    x_max = np.max(sample[feature_key])
    an_score = score_strategy(sample)
    plt.hist2d((sample[feature_key]), an_score, range=(
     (
      x_min * 0.9, np.percentile(sample[feature_key], 99.99900000000001)), (np.min(an_score), np.percentile(an_score, 99.99))),
      norm=(LogNorm()),
      bins=100)
    xs = np.arange(x_min, x_max, 0.001 * (x_max - x_min))
    plt.plot(xs, (discriminator.predict([xs, xs])), '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 & L2 > LT')
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.colorbar()
    plt.legend(loc='best')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    if fig_dir:
        plt.savefig(fig_dir + '/discriminator_cut' + plot_suffix + '.png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png((buf.getvalue()), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class PlotCutCallback(tf.keras.callbacks.Callback):

    def __init__(self, tb_plot_dir, qcd_sample, score_strategy):
        super(PlotCutCallback, self).__init__()
        self.tb_plot_dir = tb_plot_dir
        self.img_file_writer = tf.summary.create_file_writer(tb_plot_dir)
        self.qcd_sample = qcd_sample
        self.score_strategy = score_strategy

    def on_epoch_end(self, epoch, logs=None):
        img = plot_discriminator_cut(self.model, self.qcd_sample, self.score_strategy)
        with self.img_file_writer.as_default():
            tf.summary.image('Training data and cut', img, step=epoch)
        self.img_file_writer.flush()


# ******************************************** #
#                    main                      #
# ******************************************** #

if __name__ == '__main__':

    train_split = 0.3
    Parameters = recordtype('Parameters', 'vae_run_n, qr_run_n, qcd_train_sample_id, qcd_test_sample_id,                             sig_sample_id, strategy_id, epochs, read_n, lr_ini, batch_sz, quantile, norm')
    params = Parameters(vae_run_n=113,
                      qr_run_n=227,
                      qcd_train_sample_id=('qcdSigAllTrain' + str(int(train_split * 100)) + 'pct'),
                      qcd_test_sample_id=('qcdSigAllTest' + str(int((1 - train_split) * 100)) + 'pct'),
                      sig_sample_id='GtoWW35naReco',
                      strategy_id='rk5_05',
                      epochs=15,
                      read_n=(int(1e5)),
                      lr_ini=0.0001,
                      batch_sz=64,
                      quantile=0.9,
                      norm='std')

    logger = log.get_logger(__name__)
    logger.info('\n' + '*' * 70 + '\n' + '\t\t\t train QR \n' + str(params) + '\n' + '*' * 70)
    tensorboard_log_dir = 'logs/tensorboard/' + str(params.qr_run_n)
    os.system('rm -rf ' + tensorboard_log_dir)
    qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_' + str(params.vae_run_n) + '/qr_run_' + str(params.qr_run_n)
    pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)
    fig_dir = 'fig/qr_run_' + str(params.qr_run_n)
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

    #****************************************#
    #           read in qcd data
    #****************************************#

    
    qcd_train_sample = pgut.read_sample(params.qcd_train_sample_id, params, 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Train_signalregion_parts')
    qcd_test_sample = pgut.read_sample(params.qcd_test_sample_id, params, 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_signalregion_parts')
    
    score_strategy = ansc.an_score_strategy_dict[params.strategy_id]
    
    x_train, x_test = qcd_train_sample['mJJ'], qcd_test_sample['mJJ']
    y_train, y_test = score_strategy(qcd_train_sample), score_strategy(qcd_test_sample)

    accuracy_bins = np.array([1199.0, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206,2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928]).astype('float')
    x_min = np.min(x_train)
    x_mu_std = (np.mean(x_train), np.std(x_train))
    accuracy_bins = (accuracy_bins - x_mu_std[0]) / x_mu_std[1]
    
    #****************************************#
    #           build model
    #****************************************#

    layers_n = 4
    nodes_n = 4
    initializer = 'he_uniform'
    regularizer = None
    activation = 'swish'
    lr_ini = 0.001
    wd_ini = 0.0001
    quant_loss = quantile_loss(params.quantile)
    ratio_metric = scnd_fini_diff_metric() #binned_quantile_dev_loss(params.quantile, accuracy_bins)

    model = build_model(quant_loss, ratio_metric, layers_n, nodes_n, lr_ini=lr_ini, wd_ini=wd_ini, activation=activation, x_mu_std=x_mu_std, x_min=x_min, initializer=initializer, norm=(params.norm))
    model.summary()

    ### setup callbacks

    tensorboard_callb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    es_callb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-08)
    img_log_dir = tensorboard_log_dir + '/plots'
    os.system('rm -rf ' + img_log_dir + '/*')
    plot_cb = PlotCutCallback(img_log_dir, qcd_test_sample, score_strategy)
    
    ### fit model

    model.fit(x=x_train, y=y_train, batch_size=(params.batch_sz), epochs=(params.epochs), shuffle=True, validation_data=(x_test, y_test),callbacks=[tensorboard_callb, es_callb, reduce_lr, plot_cb])
    
    plot_log_transformed_results(model, x_train, y_train, fig_dir)

    ### save model
    
    model_str = stco.make_qr_model_str(run_n_qr=(params.qr_run_n), run_n_vae=(params.vae_run_n), quantile=(params.quantile), sig_id=(params.sig_sample_id), sig_xsec=0, strategy_id=(params.strategy_id))
    log.info('saving model to ' + model_str)
    model.save(os.path.join(qr_model_dir, model_str))

    ### write final cut plot

    img_file_writer = tf.summary.create_file_writer(img_log_dir)
    img = plot_discriminator_cut(model, qcd_test_sample, score_strategy, fig_dir=fig_dir)
    with img_file_writer.as_default():
        tf.summary.image('Training data and cut', img, step=1000)
# okay decompiling qr_playground_model.cpython-36.pyc
