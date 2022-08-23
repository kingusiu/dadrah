import os
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from sklearn import preprocessing
import pathlib

import dadrah.playground.playground_util as pgut
import dadrah.selection.anomaly_score_strategy as ansc
import dadrah.util.logging as log



# ******************************************** #
#             loss functions                   #
# ******************************************** #

class quantile_loss(tf.keras.losses.Loss):

    def __init__(self, quantile, name='quantileLoss'):
        super().__init__(name=name)
        self.quantile = tf.constant(quantile)

    @tf.function
    def call(self, targets, predictions):
        targets = tf.squeeze(targets)
        predictions = tf.squeeze(predictions)
        err = tf.subtract(targets, predictions)
        return tf.reduce_mean(tf.where(err>=0, self.quantile*err, (self.quantile-1)*err)) # return mean over samples in batch


# deviance of global (unbinned) ratio of above/below number of events from quantile (1 value for full batch)
class ratio_dev_loss():

    def __init__(self, quantile, name='quantileAccLoss'):
        self.name=name
        self.quantile = tf.constant(quantile)

    @tf.function
    def __call__(self, inputs, targets, predictions):
        predictions = tf.squeeze(predictions)
        count_tot = tf.shape(inputs)[0] # get batch size
        count_above = tf.math.count_nonzero(tf.math.greater(targets,predictions))
        ratio = tf.math.divide_no_nan(tf.cast(count_above,tf.float32),tf.cast(count_tot,tf.float32))
        return tf.math.square(self.quantile-ratio)


# deviance of binned ratio of below/above number of events from quantile (1 value for full batch)
class binned_quantile_dev_loss():

    def __init__(self, quantile, bins, name='binnedQuantileDevLoss'):
        self.name=name
        self.quantile = tf.constant(quantile)
        self.bins_n = len(bins)
        self.bins = tf.constant(bins.astype('float32')) # instead of squeezing and reshaping expand dims of bins to (4,1)?

    # @tf.function
    def __call__(self, inputs, targets, predictions):
        # import ipdb; ipdb.set_trace()
        predictions = tf.squeeze(predictions)
        bin_idcs = tf.searchsorted(self.bins,inputs)

        ratios = tf.Variable([self.quantile]*self.bins_n)
        for bin_idx in range(1,self.bins_n+1):
            bin_mask = tf.math.equal(bin_idcs, bin_idx)
            count_tot = tf.math.count_nonzero(bin_mask)
            # sum only when count_tot > 0! (TODO: or > 1 s.t. a ratio is even computable?)
            if count_tot > 0:
                count_above = tf.math.count_nonzero(targets[bin_mask] > predictions[bin_mask])
                ratio = tf.math.divide_no_nan(tf.cast(count_above,tf.float32),tf.cast(count_tot,tf.float32))
                ratios[bin_idx-1].assign(ratio)

        return tf.reduce_sum(tf.math.square(ratios-self.quantile)) # sum over m bins 


# ******************************************** #
#                    model                     #
# ******************************************** #


class QrModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):

        self.regularizer = kwargs.pop('regularizer', None)
        
        super().__init__(*args, **kwargs)

    def compile(self, optimizer, quant_loss, ratio_loss=None, run_eagerly=True):

        super().compile(optimizer=optimizer,run_eagerly=run_eagerly)

        self.quant_loss_fn = quant_loss
        self.ratio_loss_fn = ratio_loss

        # set up loss tracking (track scalar metric value with Mean)
        self.total_loss_mean = tf.keras.metrics.Mean('total_loss') # main quantile loss
        self.quant_loss_mean = tf.keras.metrics.Mean(self.quant_loss_fn.name)
        if self.ratio_loss_fn is not None:
            self.ratio_loss_mean = tf.keras.metrics.Mean(self.ratio_loss_fn.name)
        if self.regularizer is not None:
            self.reg_loss_mean = tf.keras.metrics.Mean('reg_loss')


    def train_step(self, data):

        inputs, targets = data
        # import ipdb; ipdb.set_trace()

        with tf.GradientTape() as tape:
            # predict
            predictions = self([inputs,targets], training=True)
            # quantile loss
            quant_loss = self.quant_loss_fn(targets, predictions)
            total_loss = quant_loss
            # regularization loss
            if self.losses:
                reg_loss = tf.math.add_n(model.losses)
                total_loss += reg_loss
            # additional optional ratio loss
            if self.ratio_loss_fn is not None:
                ratio_loss = self.ratio_loss_fn(inputs,targets,predictions) # one value per batch
                total_loss += ratio_loss
        
        trainable_variables = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        # update state of metrics for each batch
        self.total_loss_mean.update_state(total_loss)
        self.quant_loss_mean.update_state(quant_loss)
        losses = {'total_loss': self.total_loss_mean.result(), 'quant_loss': self.quant_loss_mean.result()}
        if self.ratio_loss_fn is not None:
            self.ratio_loss_mean.update_state(ratio_loss)
            losses['ratio_loss'] = self.ratio_loss_mean.result()
        if self.regularizer is not None:
            self.reg_loss_mean.update_state(reg_loss)
            losses['reg_loss'] = self.reg_loss_mean.result() 
        
        return losses


    def test_step(self, data):

        inputs, targets = data
        predictions = self([inputs,targets], training=False)
        # quantile loss
        quant_loss = self.quant_loss_fn(targets, predictions)
        total_loss = quant_loss
        # additional optional ratio loss
        if self.ratio_loss_fn is not None:
            ratio_loss = self.ratio_loss_fn(inputs,targets,predictions) # one value per batch
            total_loss += ratio_loss

        # update state of metrics
        self.quant_loss_mean.update_state(quant_loss)
        self.total_loss_mean.update_state(total_loss)
        losses = {'total_loss': self.total_loss_mean.result(), 'quant_loss': self.quant_loss_mean.result()}
        if self.ratio_loss_fn is not None:
            self.ratio_loss_mean.update_state(ratio_loss)
            losses['ratio_loss'] = self.ratio_loss_mean.result()

        return losses


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = super().metrics
        metrics.append(self.total_loss_mean)
        metrics.append(self.quant_loss_mean)
        if self.ratio_loss_fn is not None:
            metrics.append(self.ratio_loss_mean)
        if self.regularizer is not None:
            metrics.append(self.reg_loss_mean)
        return metrics



def make_model(n_layers, n_nodes, initializer='glorot_uniform', regularizer=None, bias_regularizer=None, activation='relu', dr_rate=0.):

    inputs_mjj = tf.keras.Input(shape=(1,), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1,), name='targets') # only needed for calculating metric because update() signature is limited to y & y_pred in keras.metrics class 
    x = inputs_mjj

    for i in range(n_layers):
        x = tf.keras.layers.Dense(n_nodes, kernel_initializer=initializer, kernel_regularizer=regularizer, activation=activation, name='dense'+str(i+1))(x)
    if dr_rate > 0: 
        x = tf.keras.layers.Dropout(dr_rate)(x)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, kernel_regularizer=None, bias_regularizer=bias_regularizer, name='dense'+str(n_layers+1))(x)

    model = QrModel(name='QR', inputs=[inputs_mjj,targets], outputs=outputs, regularizer=regularizer)

    return model


# ******************************************** #
#       discriminator analysis plots           #
# ******************************************** #

def plot_discriminator_cut(discriminator, sample, score_strategy, norm_x, norm_y, feature_key='mJJ', plot_name='discr_cut', fig_dir=None):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key]) #norm_x.data_min_[0] #
    x_max = np.max(sample[feature_key]) #norm_x.data_max_[0] #
    an_score = score_strategy(sample)
    plt.hist2d(sample[feature_key], an_score,
           range=((x_min*0.9 , np.percentile(sample[feature_key], 99.99)), (np.min(an_score), np.percentile(an_score, 1e2*(1-1e-4)))), 
           norm=LogNorm(), bins=100)

    xs = np.arange(x_min, x_max, 0.001*(x_max-x_min))
    xs_t = norm_x.transform(xs.reshape(-1,1)).squeeze()
    #import ipdb; ipdb.set_trace()
    y_hat = discriminator.predict([xs_t,xs_t]) # passing in dummy target
    plt.plot(xs, norm_y.inverse_transform(y_hat.reshape(-1,1)).squeeze() , '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 & L2 > LT')
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.colorbar()
    plt.legend(loc='best')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    if fig_dir:
        plt.savefig(fig_dir+'/discriminator_cut.png')
    plt.close(fig)
    return image




# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':

    train_split = 0.3
    # tf.random.set_seed(42)
    # tf.keras.utils.set_random_seed(42) # also sets python and numpy random seeds

    Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, epochs, read_n, l2_coeff, dr_rate, learn_rate, batch_sz')
    params = Parameters(
                    vae_run_n=113,
                    qr_run_n=177,
                    qcd_train_sample_id='qcdSigAllTrain'+str(int(train_split*100))+'pct', 
                    qcd_test_sample_id='qcdSigAllTest'+str(int((1-train_split)*100))+'pct',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    epochs=50,
                    read_n=int(1e6),
                    l2_coeff=1e-5,
                    dr_rate=0.,
                    learn_rate=1e-3,
                    batch_sz=1024
                    )

    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t train QR \n'+str(params)+'\n'+'*'*70)
    # tensorboard logging
    tensorboard_log_dir = 'logs/tensorboard/' + str(params.qr_run_n)
    # remove previous tensorboard logs
    os.system('rm -rf '+tensorboard_log_dir)

    tf.debugging.experimental.enable_dump_debug_info(tensorboard_log_dir+'/debug', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

    quantiles = [0.3, 0.5, 0.7, 0.9]
    # quantiles = [0.9]
    sig_xsec = 0.

    ### paths ###

    # data inputs (mjj & vae-scores): /eos/user/k/kiwoznia/data/VAE_results/events/run_$vae_run_n$
    # model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$
    # data outputs (selections [0/1] per quantile): /eos/user/k/kiwoznia/data/QR_results/events/

    qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)
    pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)

    #****************************************#
    #           read in qcd data
    #****************************************#

    qcd_train_sample = pgut.read_sample(params.qcd_train_sample_id, params, 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Train_signalregion_parts')
    qcd_test_sample = pgut.read_sample(params.qcd_test_sample_id, params, 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_signalregion_parts')

    #****************************************#
    #           prepare inputs & targets
    #****************************************#
    score_strategy = ansc.an_score_strategy_dict[params.strategy_id]
    x_train, x_test = qcd_train_sample['mJJ'], qcd_test_sample['mJJ']
    y_train, y_test = score_strategy(qcd_train_sample), score_strategy(qcd_test_sample) 

    # set up min max normalizations
    norm_x_mima = preprocessing.MinMaxScaler()
    norm_y_mima = preprocessing.MinMaxScaler()

    x_train_mima, x_test_mima = norm_x_mima.fit_transform(x_train.reshape(-1,1)).squeeze(), norm_x_mima.transform(x_test.reshape(-1,1)).squeeze()
    y_train_mima, y_test_mima = norm_y_mima.fit_transform(y_train.reshape(-1,1)).squeeze(), norm_y_mima.transform(y_test.reshape(-1,1)).squeeze()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_mima, y_train_mima))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_mima, y_test_mima))

    train_dataset = train_dataset.shuffle(int(1e6)).batch(params.batch_sz)
    test_dataset = test_dataset.batch(params.batch_sz) 

    # normalize bin edges for accuracy metric
    # accuracy_bins = np.array([1199.,2200.,3200.,4200.]) # min-max normalized below!
    # accuracy_bins = np.array([1199.,2000.,3000.,4000.]) # min-max normalized below!
    # accuracy_bins = np.array([1199.,1400.,1600.,1800.]) # min-max normalized below!
    accuracy_bins = np.array([1199., 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928]).astype('float')

    accuracy_bins = norm_x_mima.transform(accuracy_bins.reshape(-1,1)).squeeze()

    # import ipdb; ipdb.set_trace()

    #****************************************#
    #           build model
    #****************************************#

    layers_n = 2
    nodes_n = 12
    quantile = 0.5
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learn_rate) # tf.keras.optimizers.SGD(learning_rate=params.learn_rate) 
    initializer = 'he_uniform'
    regularizer = None # tf.keras.regularizers.l2(l2=params.l2_coeff) # -> use only with SGD!
    activation = 'elu'
    quant_loss = quantile_loss(quantile)
    ratio_loss = binned_quantile_dev_loss(quantile, accuracy_bins) #None

    model = make_model(layers_n, nodes_n, initializer=initializer, regularizer=regularizer, activation=activation, dr_rate=params.dr_rate)
    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    # import ipdb; ipdb.set_trace()
    model.compile(optimizer=optimizer, quant_loss=quant_loss, ratio_loss=ratio_loss, run_eagerly=True)
    model.fit(x=x_train_mima, y=y_train_mima, batch_size=params.batch_sz, epochs=params.epochs, validation_data=(x_test_mima, y_test_mima), callbacks=[tensorboard_callback])

    # print image to tensorboard
    img_log_dir = tensorboard_log_dir + '/plots'
    os.system('rm -rf '+img_log_dir+'/*')
    img_file_writer = tf.summary.create_file_writer(img_log_dir)
    img = plot_discriminator_cut(model, qcd_test_sample, score_strategy, norm_x_mima, norm_y_mima)
    with img_file_writer.as_default():
        tf.summary.image("Training data and cut", img, step=0)