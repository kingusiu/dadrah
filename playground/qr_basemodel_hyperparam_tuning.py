import os
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
import tensorflow_addons as tfa
import kerastuner as kt
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
import vande.vae.layers as layers



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



# deviance of global (unbinned) ratio of above/below number of events from quantile (1 value for full batch)
class quantile_dev_loss():

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
                count_bel = tf.math.count_nonzero(targets[bin_mask] < predictions[bin_mask])
                ratio = tf.math.divide_no_nan(tf.cast(count_bel,tf.float32),tf.cast(count_tot,tf.float32))
                ratios[bin_idx-1].assign(ratio)

        return tf.reduce_sum(tf.math.square(ratios-self.quantile)) # sum over m bins 


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



# ******************************************** #
#                    model                     #
# ******************************************** #

class QrModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        (super().__init__)(*args, **kwargs)

    def compile(self, loss, metric_fn, optimizer, run_eagerly=True, **kwargs):
        (super().compile)(optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)
        self.quant_loss_fn = loss
        self.metric_fn = metric_fn
        self.loss_mean = tf.keras.metrics.Mean('loss')
        self.ratio_metric_mean = tf.keras.metrics.Mean(self.metric_fn.name)

    def train_step(self, data):

        inputs, targets = data
        
        with tf.GradientTape() as (tape):
            predictions = self([inputs, targets], training=True)
            loss = self.quant_loss_fn(targets, predictions)
        
        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        if self.metric_fn.name == '2ndDiff':
            delta = self.metric_fn.delta
            # import ipdb; ipdb.set_trace()
            pred = self([inputs, targets], training=False) # compute new predictions after backpropagation step
            pred_delta_plus = self([inputs+delta, targets], training=False)
            pred_delta_minus = self([inputs-delta, targets], training=False)
            norm_delta = tf.math.divide_no_nan(delta,self.get_layer('Normalization').std_x) # scale delta like inputs
            metric_val = self.metric_fn(pred, pred_delta_plus, pred_delta_minus, norm_delta)

        else:
            inputs_norm = self.get_layer('Normalization')(inputs)
            metric_val = self.metric_fn(inputs_norm, targets, predictions)

        self.loss_mean.update_state(loss)
        self.ratio_metric_mean.update_state(metric_val)

        return {'loss':self.loss_mean.result(), self.metric_fn.name : self.ratio_metric_mean.result()}


    def test_step(self, data):

        inputs, targets = data

        predictions = self([inputs, targets], training=False)

        loss = self.quant_loss_fn(targets, predictions)

        if self.metric_fn.name == '2ndDiff':
            delta = self.metric_fn.delta
            # import ipdb; ipdb.set_trace()
            pred_delta_plus = self([inputs+delta, targets], training=False)
            pred_delta_minus = self([inputs-delta, targets], training=False)
            norm_delta = tf.math.divide_no_nan(delta,self.get_layer('Normalization').std_x)
            metric_val = self.metric_fn(predictions, pred_delta_plus, pred_delta_minus, norm_delta)

        else:
            inputs_norm = self.get_layer('Normalization')(inputs)
            metric_val = self.metric_fn(inputs_norm, targets, predictions)

        self.loss_mean.update_state(loss)
        self.ratio_metric_mean.update_state(metric_val)
        
        return {'loss':self.loss_mean.result(), self.metric_fn.name : self.ratio_metric_mean.result()}

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_mean)
        metrics.append(self.ratio_metric_mean)
        return metrics


    @classmethod
    def from_config(cls, config):
        return cls(**config)



def build_model_with_hp(hp, quantile_loss, metric_fn, x_mu_std=(0.,1.), initializer='glorot_uniform'):

    # sample hyperparameters
    layers_n = hp.Int(name='layers_n',min_value=1,max_value=6)
    nodes_n = hp.Int(name='nodes_n',min_value=4,max_value=16)
    # optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    lr_ini = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    # wd_ini = hp.Float('weight_decay', min_value=1e-6, max_value=1e-3, sampling='log')
    # lr_schedule = tf.optimizers.schedules.ExponentialDecay(lr_ini, 10000, 0.97)
    # wd_schedule = tf.optimizers.schedules.ExponentialDecay(wd_ini, 10000, 0.97)
    # optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=lambda:None)
    # optimizer = tfa.optimizers.AdamW(learning_rate=lr_ini, weight_decay=wd_ini)
    # optimizer.weight_decay = lambda : wd_schedule(optimizer.iterations)
    # if optimizer == "sgd":
    #     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9) # todo: add momentum
    #     reg_val = hp.Float(name='regularizer',min_value=1e-6,max_value=1e-3) # regularize weights when using SGD
    #     regularizer = tf.keras.regularizers.l2(l2=reg_val)
    # else:
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #     regularizer = None
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_ini)
    activation = hp.Choice('activation', values=['elu', 'swish'])

    # model architecture
    inputs_mjj = tf.keras.Input(shape=(1,), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1,), name='targets') # only needed for calculating metric because update() signature is limited to y & y_pred in keras.metrics class 
    x = inputs_mjj
    x = layers.StdNormalization(*x_mu_std,name='Normalization')(x)
    for _ in range(layers_n):
        x = tf.keras.layers.Dense(nodes_n, kernel_initializer=initializer, activation=activation)(x)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
    model = QrModel(inputs=[inputs_mjj,targets], outputs=outputs)
    model.compile(loss=quantile_loss, metric_fn=metric_fn, optimizer=optimizer) # Adam(lr=1e-3) TODO: add learning rate

    return model


class QrHyperparamModel(kt.HyperModel):

    def __init__(self, quantile_loss, metric_fn, x_mu_std):
        self.quantile_loss = quantile_loss
        self.metric_fn = metric_fn
        self.x_mu_std = x_mu_std

    def build(self, hp):
        return build_model_with_hp(hp, self.quantile_loss, self.metric_fn, self.x_mu_std)

    def fit(self, hp, model, x, y, **kwargs):
        # import ipdb; ipdb.set_trace()
        batch_sz = hp.Choice(name='batch_sz',values=[32,128,256,512,1024,2048,4096])
        print('batch_sz: ' + str(batch_sz))
        return model.fit(x, y, batch_size=batch_sz, **kwargs)


# learning rate printer callback
class PrintLearningRate(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        # lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        print("\nLearning rate at epoch {} is {:.3e}".format(epoch, lr))


# learning rate decay and weight decay
class DecayHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.lr = []
        self.wd = []
    def on_batch_end(self, batch, logs={}):
        self.lr.append(self.model.optimizer.lr(self.model.optimizer.iterations))
        self.wd.append(self.model.optimizer.weight_decay)


class WeightDecayLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs["weight_decay"] = self.model.optimizer.weight_decay

# ******************************************** #
#       discriminator analysis plots           #
# ******************************************** #

def plot_discriminator_cut(discriminator, sample, score_strategy, feature_key='mJJ', plot_name='discr_cut', fig_dir=None, plot_suffix='',xlim=True):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])
    x_max = np.max(sample[feature_key])
    an_score = score_strategy(sample)
    x_top = np.percentile(sample[feature_key], 99.999) if xlim else x_max
    x_range = ((x_min * 0.9,x_top), (np.min(an_score), np.percentile(an_score, 99.99)))
    plt.hist2d((sample[feature_key]), an_score, range=x_range, norm=(LogNorm()),bins=100)
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
            tf.summary.image("Training data and cut", img, step=epoch)
        self.img_file_writer.flush()



# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':

    train_split = 0.3
    # tf.random.set_seed(42)
    # tf.keras.utils.set_random_seed(42) # also sets python and numpy random seeds

    Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_train_sample_id, qcd_test_sample_id, \
        sig_sample_id, strategy_id, epochs, read_n, objective, max_trials, quantile')
    params = Parameters(
                    vae_run_n=113,
                    qr_run_n=237,
                    qcd_train_sample_id='qcdSigAllTrain'+str(int(train_split*100))+'pct', 
                    qcd_test_sample_id='qcdSigAllTest'+str(int((1-train_split)*100))+'pct',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    epochs=50,
                    read_n=int(5e5),
                    objective='val_2ndDiff',
                    max_trials=20,
                    quantile=0.5
                    )

    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t train QR \n'+str(params)+'\n'+'*'*70)
    # tensorboard logging
    tensorboard_log_dir = 'logs/tensorboard/' + str(params.qr_run_n)
    # remove previous tensorboard logs
    os.system('rm -rf '+tensorboard_log_dir)

    sig_xsec = 0.

    ### paths ###

    # data inputs (mjj & vae-scores): /eos/user/k/kiwoznia/data/VAE_results/events/run_$vae_run_n$
    # model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$
    # data outputs (selections [0/1] per quantile): /eos/user/k/kiwoznia/data/QR_results/events/

    qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)
    pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)
    fig_dir = 'fig/qr_run_'+str(params.qr_run_n)
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)


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
    # norm_x_mima = preprocessing.MinMaxScaler()
    # norm_y_mima = preprocessing.MinMaxScaler()

    # x_train_mima, x_test_mima = norm_x_mima.fit_transform(x_train.reshape(-1,1)).squeeze(), norm_x_mima.transform(x_test.reshape(-1,1)).squeeze()
    # y_train_mima, y_test_mima = norm_y_mima.fit_transform(y_train.reshape(-1,1)).squeeze(), norm_y_mima.transform(y_test.reshape(-1,1)).squeeze()

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train_mima, y_train_mima))
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test_mima, y_test_mima))

    # train_dataset = train_dataset.shuffle(int(1e6)).batch(params.batch_sz)
    # test_dataset = test_dataset.batch(params.batch_sz) 

    # normalize bin edges for accuracy metric
    # accuracy_bins = np.array([1199.,2200.,3200.,4200.]) # min-max normalized below!
    # accuracy_bins = np.array([1199.,1900.,2900.,3900.]) # min-max normalized below!
    # accuracy_bins = np.array([1199.,1400.,1600.,1800.]) # min-max normalized below!
    accuracy_bins = np.array([1199., 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928]).astype('float')

    x_mu_std = (np.mean(x_train), np.std(x_train))
    accuracy_bins = (accuracy_bins-x_mu_std[0])/x_mu_std[1]

    # accuracy_bins = norm_x_mima.transform(accuracy_bins.reshape(-1,1)).squeeze()

    # import ipdb; ipdb.set_trace()

    #****************************************#
    #           build model
    #****************************************#

    initializer = 'he_uniform'
    quant_loss = quantile_loss_smooth(params.quantile) #quantile_loss(params.quantile)
    metric_fn = scnd_fini_diff_metric() #binned_quantile_dev_loss(params.quantile, accuracy_bins)

    logger.info('loss fun ' + quant_loss.name + ', metric fun ' + metric_fn.name)

    # hyperparams tuner
    objective = kt.Objective(name=params.objective, direction='min')
    tuner = kt.BayesianOptimization(QrHyperparamModel(quant_loss, metric_fn, x_mu_std=x_mu_std), 
                            objective=objective, max_trials=params.max_trials, overwrite=True, 
                            directory='logs',project_name='bayes_tune_'+str(params.qr_run_n))

    tensorboard_callb = tf.keras.callbacks.TensorBoard(log_dir=tuner.project_dir+'/tensorboard', histogram_freq=1)
    es_callb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-9) # only with sgd!?

    tuner.search(x=x_train, y=y_train, epochs=params.epochs, shuffle=True,
            validation_data=(x_test, y_test), callbacks=[tensorboard_callb, es_callb, reduce_lr, PrintLearningRate()])


    best_trials = tuner.oracle.get_best_trials(num_trials=3)
    for best_trial in best_trials:
        best_trial.summary()

    top_models = tuner.get_best_models(num_models=3)
    img_log_dir = tuner.project_dir+'/tensorboard/plots'
    os.system('rm -rf '+img_log_dir+'/*')
    img_file_writer = tf.summary.create_file_writer(img_log_dir)
    for i, top_model in enumerate(top_models):
        # print image to tensorboard
        img = plot_discriminator_cut(top_model, qcd_test_sample, score_strategy, fig_dir=fig_dir, plot_suffix='_model'+str(i))
        with img_file_writer.as_default():
            tf.summary.image("Training data and cut model " +str(i), img, step=1000)
        img_file_writer.flush() 
    img_file_writer.close()
