import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
import numpy as np
from recordtype import recordtype
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import pathlib 
import dadrah.playground.playground_util as pgut 
import dadrah.selection.anomaly_score_strategy as ansc 
import dadrah.util.logging as log 
import dadrah.util.string_constants as stco
import dadrah.selection.quantile_regression as qure
import vande.vae.layers as layers



###################################
#       losses & metrics


class quantile_loss_smooth(tf.keras.losses.Loss):
    ''' S. Zheng, “Gradient descent algorithms for quantile regression with
    smooth approximation,” International Journal of Machine Learning and
    Cybernetics, vol. 2, no. 3, p. 191, 2011. '''

    def __init__(self, quantile, name='quantileLossSmooth'):
        super().__init__(name=name)
        self.quantile = tf.constant(quantile)
        self.beta = tf.constant(0.2) # smoothing constant

    @tf.function
    def call(self, targets, predictions):
        targets = tf.squeeze(targets)
        predictions = tf.squeeze(predictions)
        err = tf.subtract(targets, predictions)
        return tf.reduce_mean(self.quantile * err + self.beta * tf.math.log(1. + tf.math.exp(-tf.math.divide_no_nan(err,self.beta)))) # return mean over samples in batch



class scnd_fini_diff_metric():

    def __init__(self, delta=1.0, name='2ndDiff'):
        self.name=name
        self.delta = tf.constant(delta) # delta to approximate second derivative, applied before normalization -> to O(1K)

    # @tf.function
    def __call__(self, pred, pred_delta_plus, pred_delta_minus, delta): # for integration in regular TF -> compute predictions for delta-shifted inputs in outside train/test step
        # import ipdb; ipdb.set_trace()
        pred = tf.squeeze(pred)
        pred_delta_plus = tf.squeeze(pred_delta_plus) # targets input not used in prediction
        pred_delta_minus = tf.squeeze(pred_delta_minus)
        
        # 2nd finite diff
        fini_diff2 = tf.math.divide_no_nan((pred_delta_plus - tf.cast(tf.constant(2.0),tf.float32)*pred + pred_delta_minus),tf.math.square(delta)) # using scaled delta here  

        return tf.reduce_mean(tf.math.square(fini_diff2)) # mean per batch



#########################
#         model


class QrModel(tf.keras.Model):


    def compile(self, loss, metric_fn, optimizer, run_eagerly=True, **kwargs):
        (super().compile)(optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)
        self.quant_loss_fn = loss
        self.metric_fn = metric_fn
        self.loss_mean = tf.keras.metrics.Mean('loss')
        self.metric_mean = tf.keras.metrics.Mean(self.metric_fn.name)

    def train_step(self, data):

        inputs, targets = data
        
        with tf.GradientTape() as (tape):
            predictions = self(inputs, training=True)
            loss = self.quant_loss_fn(targets, predictions)
        
        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        delta = self.metric_fn.delta
        # import ipdb; ipdb.set_trace()
        pred = self(inputs, training=False) # compute new predictions after backpropagation step
        pred_delta_plus = self(inputs+delta, training=False)
        pred_delta_minus = self(inputs-delta, training=False)
        norm_delta = tf.math.divide_no_nan(delta,self.get_layer('Normalization').std_x) # scale delta like inputs
        metric_val = self.metric_fn(pred, pred_delta_plus, pred_delta_minus, norm_delta)

        self.loss_mean.update_state(loss)
        self.metric_mean.update_state(metric_val)

        return {'loss':self.loss_mean.result(), self.metric_fn.name : self.metric_mean.result()}


    def test_step(self, data):

        inputs, targets = data

        predictions = self(inputs, training=False)

        loss = self.quant_loss_fn(targets, predictions)

        delta = self.metric_fn.delta
        # import ipdb; ipdb.set_trace()
        pred_delta_plus = self(inputs+delta, training=False)
        pred_delta_minus = self(inputs-delta, training=False)
        norm_delta = tf.math.divide_no_nan(delta,self.get_layer('Normalization').std_x)
        metric_val = self.metric_fn(predictions, pred_delta_plus, pred_delta_minus, norm_delta)


        self.loss_mean.update_state(loss)
        self.metric_mean.update_state(metric_val)
        
        return {'loss':self.loss_mean.result(), self.metric_fn.name : self.metric_mean.result()}

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_mean)
        metrics.append(self.metric_mean)
        return metrics



def build_model(loss_fn, metric_fn, layers_n, nodes_n, lr_ini, wd_ini, activation, x_mu_std=(0.0, 1.0), initializer='he_uniform'):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_ini)

    inputs_mjj = tf.keras.Input(shape=(1, ), name='inputs_mjj')
    
    x = inputs_mjj
    x = layers.StdNormalization(*x_mu_std, name='Normalization')(x)
    for i in range(layers_n):
        x = tf.keras.layers.Dense(nodes_n, kernel_initializer=initializer, activation=activation, name='dense_'+str(i+1))(x)

    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, name='dense_out')(x)
    
    model = QrModel(inputs=inputs_mjj, outputs=outputs)
    model.compile(loss=loss_fn, metric_fn=metric_fn, optimizer=optimizer)
    
    return model



def plot_discriminator_cut(discriminator, sample, score_strategy, feature_key='mJJ', plot_name='discr_cut', fig_dir=None, plot_suffix='', xlim=False):
    
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])
    x_max = np.max(sample[feature_key])
    an_score = score_strategy(sample)
    plt.hist2d((sample[feature_key]), an_score, norm=(LogNorm()),bins=100)
    xs = np.arange(x_min, x_max, 0.001 * (x_max - x_min))
    plt.plot(xs, discriminator.predict(xs), '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 & L2 > LT')
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.colorbar()
    plt.legend(loc='best')
    if fig_dir:
        plt.savefig(fig_dir + '/discriminator_cut' + plot_suffix + '.png')
    plt.close(fig)
    

# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':

    train_split = 0.3
    Parameters = recordtype('Parameters', 'vae_run_n, qr_run_n, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, epochs, read_n, lr_ini, batch_sz, quantile, norm')
    params = Parameters(vae_run_n=113,
                      qr_run_n=5,
                      qcd_train_sample_id=('qcdSigAllTrain' + str(int(train_split * 100)) + 'pct'),
                      qcd_test_sample_id=('qcdSigAllTest' + str(int((1 - train_split) * 100)) + 'pct'),
                      sig_sample_id='GtoWW35naReco',
                      strategy_id='rk5_05',
                      epochs=3,
                      read_n=(int(1e3)),
                      lr_ini=0.0001,
                      batch_sz=256,
                      quantile=0.3,
                      norm='std')

    logger = log.get_logger(__name__)
    logger.info('\n' + '*' * 70 + '\n' + '\t\t\t train QR \n' + str(params) + '\n' + '*' * 70)
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

    x_min = np.min(x_train)
    x_max = np.max(x_train)
    x_mu_std = (np.mean(x_train), np.std(x_train))
    
    #****************************************#
    #           build model
    #****************************************#

    layers_n = 1
    nodes_n = 4
    initializer = 'he_uniform'
    regularizer = None
    activation = 'swish'
    lr_ini = 0.001
    wd_ini = 0.0001
    quant_loss = quantile_loss_smooth(params.quantile) #quantile_loss(params.quantile)
    ratio_metric = scnd_fini_diff_metric() #binned_quantile_dev_loss(params.quantile, accuracy_bins)

    logger.info('loss fun ' + quant_loss.name + ', metric fun ' + ratio_metric.name)

    model = build_model(quant_loss, ratio_metric, layers_n, nodes_n, lr_ini=lr_ini, wd_ini=wd_ini, activation=activation, x_mu_std=x_mu_std, initializer=initializer)
    model.summary()

    ### fit model

    model.fit(x=x_train, y=y_train, batch_size=params.batch_sz, epochs=params.epochs, shuffle=True, validation_data=(x_test, y_test))
    
    ### save model
    
    model_str = stco.make_qr_model_str(run_n_qr=(params.qr_run_n), run_n_vae=(params.vae_run_n), quantile=(params.quantile), sig_id=(params.sig_sample_id), sig_xsec=0, strategy_id=(params.strategy_id))
    model_full_path = os.path.join(qr_model_dir, model_str)
    logger.info('saving model to ' + model_str)
    model.save(model_full_path)

    ### test both models on toy inputs
    xs = np.arange(x_min, x_max, 0.01 * (x_max - x_min))
    orig_pred = np.squeeze(model.predict(xs))
    orig_weights = model.get_weights()

    plot_discriminator_cut(model, qcd_test_sample, score_strategy, fig_dir=fig_dir, plot_suffix='original', xlim=False)

    ### delete and reload model
    del model
    loaded_model = tf.keras.models.load_model(model_full_path, custom_objects={'QrModel': QrModel, 'StdNormalization': layers.StdNormalization}, compile=False)
    loaded_pred = np.squeeze(loaded_model.predict(xs))
    loaded_weights = loaded_model.get_weights()

    # import ipdb; ipdb.set_trace()
    # test for equivalence
    assert np.allclose(orig_pred, loaded_pred)

    print(orig_weights)
    print('\n')
    print(loaded_weights)


    ### write final cut plot

    plot_discriminator_cut(loaded_model, qcd_test_sample, score_strategy, fig_dir=fig_dir, plot_suffix='loaded', xlim=False)


