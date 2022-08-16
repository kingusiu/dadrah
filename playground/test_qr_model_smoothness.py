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
from collections import OrderedDict

import pofah.jet_sample as jesa
import dadrah.util.run_paths as runpa
import dadrah.selection.anomaly_score_strategy as ansc
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.selection.qr_workflow as qrwf



# ******************************************** #
#           quantile regression loss           #
# ******************************************** #

def quantile_loss_fun(quantile):
    @tf.function
    def loss(target, pred):
        err = tf.subtract(target, pred)
        return tf.where(err>=0, quantile*err, (quantile-1)*err)
    return loss

class quantile_loss(tf.keras.losses.Loss):

    def __init__(self, quantile, name='quantileLoss'):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, target, pred):
        err = tf.subtract(target, pred)
        return tf.where(err>=0, self.quantile*err, (self.quantile-1)*err)


# deviance of global (unbinned) ratio of below/above number of events from quantile (1 value for full batch)
class quantile_accuracy_loss():

    def __init__(self, quantile, name='quantileAccLoss'):
        self.name=name
        self.quantile = quantile

    def __call__(self, inputs, targets, predictions):
        predictions = tf.squeeze(predictions)
        count_tot = tf.shape(inputs)[0] # get batch size
        count_above = tf.math.count_nonzero(targets > predictions)
        ratio = tf.math.divide_no_nan(tf.cast(count_above,tf.float32),tf.cast(count_tot,tf.float32))
        return tf.math.square(self.quantile-ratio)


# accuracy_bins = np.array([1199.,2200.,3200.,4200.]) # min-max normalized below!
accuracy_bins = np.array([1199.,2000.,3000.,4000.]) # min-max normalized below!

# deviance of binned ratio of below/above number of events from quantile (1 value for full batch)
class binned_quantile_accuracy_loss():

    def __init__(self, quantile, name='binnedQuantileAccLoss'):
        self.name=name
        self.quantile = quantile

    def __call__(self, inputs, targets, predictions):
        
        predictions = tf.squeeze(predictions)
        bins = tf.constant(accuracy_bins.astype('float32')) # instead of squeezing and reshaping expand dims of bins to (4,1)?
        bin_idcs = tf.searchsorted(bins,inputs)

        ratios = []
        for bin_idx in range(1,len(accuracy_bins)+1):
            count_tot = tf.math.count_nonzero(inputs[bin_idcs==bin_idx])
            # sum only when count_tot > 0!
            if count_tot > 0:
                count_above = tf.math.count_nonzero(targets[bin_idcs==bin_idx] > predictions[bin_idcs==bin_idx])
                ratio = tf.math.divide_no_nan(tf.cast(count_above,tf.float32),tf.cast(count_tot,tf.float32))
                ratios.append(ratio)
        ratios = tf.convert_to_tensor(ratios)

        return tf.reduce_sum(tf.math.square(tf.constant(self.quantile)-ratios)) # sum over 4 bins 

### accuracy metric 

def quantile_accuracy_wrap(inputs, targets, predictions, bin_idx, part): 

    @tf.function
    def quantile_accuracy(inputs, targets, predictions): # [batch_size x 1, batch_size x 1, batch_size x 1]
        # one accuracy metric for each bin (keras logistics): calculate target above or below prediction

        # todo: compute divergence from quantile instead of percentage (need to pass in quantile as well) -> could be used as part of loss
        # squeeze predictions' dangling 1-dimension
        predictions = tf.squeeze(predictions)
        # divide into 4 bins on mjj
        bins = tf.constant(accuracy_bins.astype('float32')) # instead of squeezing and reshaping expand dims of bins to (4,1)?
        bin_idcs = tf.searchsorted(bins,inputs)
        # bin_idcs = tf.reshape(bin_idcs, shape=(tf.shape(inputs)[0],1)) # reshape to batch size

        # import ipdb; ipdb.set_trace()
        # reduces to one value per batch (how many samples below/above cut in this batch)
        count_tot = tf.math.count_nonzero(inputs[bin_idcs==bin_idx])
        if part == 'ab':
            count = tf.math.count_nonzero(targets[bin_idcs==bin_idx] > predictions[bin_idcs==bin_idx])
        else:
            count = tf.math.count_nonzero(targets[bin_idcs==bin_idx] <= predictions[bin_idcs==bin_idx]) # check for security also below
        # return share of total events per bin
        return count_tot, tf.math.divide_no_nan(tf.cast(count,tf.float32),tf.cast(count_tot,tf.float32))

    return quantile_accuracy


class Custom_Train_Step_Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):

        # pop arguments not expected by keras.Model class. Not the nicest solution -> TODO: nice up
        self.custom_loss = kwargs.pop('custom_loss', None)

        super().__init__(*args, **kwargs)

        inputs_mjj, targets = kwargs['inputs']
        outputs = kwargs['outputs']

        if self.custom_loss is not None:
            self.custom_loss_mean = tf.keras.metrics.Mean(custom_loss.name)

        # add custom metrics
        self.custom_metric_means = {}

        self.custom_metric_funs = {}
        # do this for 4 bins, passing in bin idx to quantile accuracy, aggregation = mean of batches in each epoch
        for bin_idx in range(1,5):
            for part in ['ab', 'bo']:
                name = 'qacc_b'+str(bin_idx)+'_'+part
                self.custom_metric_funs[name] = quantile_accuracy_wrap(inputs_mjj,targets,outputs,bin_idx=bin_idx,part=part)
                self.custom_metric_means[name] = tf.keras.metrics.Mean(name) # collecting every batch result in mean (batches per epoch) aggregator

    
    def train_step(self, data):

        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self([inputs,targets], training=True)
            quantile_loss = self.compiled_loss(targets, predictions, regularization_losses=self.losses)
            loss = quantile_loss
            if self.custom_loss is not None:
                ratio_loss = self.custom_loss(inputs,targets,predictions) # one value per batch
                self.custom_loss_mean.update_state(ratio_loss) # keeps track of mean over batches
                loss += ratio_loss

        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        # import ipdb; ipdb.set_trace()
        # update state of metrics
        for name, metric_fun in self.custom_metric_funs.items():
            bin_count, metric_per_batch = metric_fun(inputs, targets, predictions)
            weight = 1 if bin_count > 0 else 0 # don't count metric if no events in that bin for this batch
            self.custom_metric_means[name].update_state(metric_per_batch, weight)

        losses_and_metrics = {'loss':loss, **{metric.name: metric.result() for metric in self.custom_metric_means.values()}}
        if self.custom_loss is not None:
            losses_and_metrics['quantile_loss'] = quantile_loss
            losses_and_metrics['ratio_loss'] = self.custom_loss_mean.result()
        return losses_and_metrics


    def test_step(self, data):

        inputs, targets = data
        predictions = self([inputs,targets], training=False)
        quantile_loss = self.compiled_loss(targets, predictions, regularization_losses=self.losses)
        loss = quantile_loss
        if self.custom_loss is not None:
            ratio_loss = self.custom_loss(inputs,targets,predictions)
            self.custom_loss_mean.update_state(ratio_loss)
            loss += ratio_loss

        losses_and_metrics = {'loss': loss}
        if self.custom_loss is not None:
            losses_and_metrics['quantile_loss'] = quantile_loss
            losses_and_metrics['ratio_loss'] = self.custom_loss_mean.result()
        return losses_and_metrics

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = super().metrics
        for custom_metric in self.custom_metric_means.values():
            metrics.append(custom_metric)
        if self.custom_loss is not None:
            metrics.append(self.custom_loss_mean)
        return metrics


def make_model(n_layers, n_nodes, initializer='glorot_uniform', regularizer=None, activation='relu', dr_rate=0., custom_train=False, custom_loss=None):

    inputs_mjj = tf.keras.Input(shape=(1,), name='inputs_mjj')
    targets = tf.keras.Input(shape=(1,), name='targets') # only needed for calculating metric because update() signature is limited to y & y_pred in keras.metrics class 
    x = inputs_mjj

    for i in range(n_layers):
        x = tf.keras.layers.Dense(n_nodes, kernel_initializer=initializer, kernel_regularizer=regularizer, activation=activation, name='dense'+str(i+1))(x)
    if dr_rate > 0: 
        x = tf.keras.layers.Dropout(dr_rate)(x)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, kernel_regularizer=None, bias_regularizer=tf.keras.regularizers.L2(1e-2), name='dense'+str(n_layers+1))(x)

    if custom_train:
        model = Custom_Train_Step_Model(name='QR', inputs=[inputs_mjj,targets], outputs=outputs, custom_loss=custom_loss)
    else:
        model = tf.keras.Model(name='QR', inputs=[inputs_mjj,targets], outputs=outputs)

    return model


# ******************************************** #
#           training and testing               #
# ******************************************** #

dense_6_bias = []

def log_weights(model, train_summary_writer, epoch):
    with train_summary_writer.as_default():
        for layer in model.layers:
          for weight in layer.weights:
            weight_name = weight.name.replace(':', '_')
            tf.summary.histogram(weight_name, weight, step=epoch)
        train_summary_writer.flush()
        # for ll in model.layers[1:]: 
        #     w, b = ll.get_weights()
        #     if len(b) == 1:
        #         dense_6_bias.append(b)
        #         tf.summary.histogram(ll.name+'/bias', b[0], step=epoch)
        #     else: 
        #         tf.summary.histogram(ll.name+'/bias', b, step=epoch) 
        #     tf.summary.histogram(ll.name+'/kernel', w, step=epoch) 


@tf.function
def train_step(model, optimizer, loss_fun, x_train, y_train, train_loss):

    # import ipdb; ipdb.set_trace()

    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_fun(y_train, predictions)

    trainable_weights = model.trainable_weights
    grads = tape.gradient(loss, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))
    # optimizer.minimize(loss, trainable_weights, tape=tape)

    train_loss(loss)

@tf.function
def test_step(model, loss_fun, x_test, y_test, test_loss):

    predictions = model(x_test)
    loss = loss_fun(y_test, predictions)

    test_loss(loss)


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
    keras_fit = True
    custom_train = True
    # tf.random.set_seed(42)
    # tf.keras.utils.set_random_seed(42) # also sets python and numpy random seeds

    Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, epochs, read_n, qr_model_t, l2_coeff, dr_rate, learn_rate, batch_sz')
    params = Parameters(
                    vae_run_n=113,
                    qr_run_n=159,
                    qcd_train_sample_id='qcdSigAllTrain'+str(int(train_split*100))+'pct', 
                    qcd_test_sample_id='qcdSigAllTest'+str(int((1-train_split)*100))+'pct',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    epochs=20,
                    read_n=int(5e5),
                    qr_model_t=stco.QR_Model.DENSE,
                    l2_coeff=1e-5,
                    dr_rate=0.,
                    learn_rate=1e-4,
                    batch_sz=512
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
    in_train_path_ext_dict = {'run': str(params.vae_run_n), 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Train_signalregion_parts': None}
    in_test_path_ext_dict = {'run': str(params.vae_run_n), 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_signalregion_parts': None}

    ### paths ###

    # data inputs (mjj & vae-scores): /eos/user/k/kiwoznia/data/VAE_results/events/run_$vae_run_n$
    # model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$
    # data outputs (selections [0/1] per quantile): /eos/user/k/kiwoznia/data/QR_results/events/

    paths_train = runpa.RunPaths(in_data_dir=stco.dir_path_dict['base_dir_vae_results_qcd_train'], in_data_names=stco.file_name_path_dict, out_data_dir=stco.dir_path_dict['base_dir_qr_selections'])
    paths_train.extend_in_path_data(in_train_path_ext_dict)
    paths_test = runpa.RunPaths(in_data_dir=stco.dir_path_dict['base_dir_vae_results_qcd_train'], in_data_names=stco.file_name_path_dict, out_data_dir=stco.dir_path_dict['base_dir_qr_selections'])
    paths_test.extend_in_path_data(in_test_path_ext_dict)

    qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_'+str(params.vae_run_n)+'/qr_run_'+str(params.qr_run_n)
    pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)

    #****************************************#
    #           read in qcd data
    #****************************************#

    qcd_train_sample = jesa.JetSample.from_input_file(params.qcd_train_sample_id, paths_train.in_file_path(params.qcd_train_sample_id), read_n=params.read_n) 
    qcd_test_sample = jesa.JetSample.from_input_file(params.qcd_test_sample_id, paths_test.in_file_path(params.qcd_test_sample_id), read_n=params.read_n)
    logger.info('read {} training samples from {}'.format(len(qcd_train_sample), paths_train.in_file_path(params.qcd_train_sample_id)))
    logger.info('read {} test samples from {}'.format(len(qcd_test_sample), paths_test.in_file_path(params.qcd_test_sample_id)))

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
    accuracy_bins = norm_x_mima.transform(accuracy_bins.reshape(-1,1)).squeeze()

    # import ipdb; ipdb.set_trace()

    #****************************************#
    #           build model
    #****************************************#

    layers_n = 2
    nodes_n = 12
    quantile = 0.5
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learn_rate) 
    initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.l2(l2=params.l2_coeff)
    activation = 'elu'
    loss_fun = quantile_loss(quantile)
    custom_loss = binned_quantile_accuracy_loss(quantile) #None

    model = make_model(layers_n, nodes_n, initializer=initializer, regularizer=regularizer, activation=activation, dr_rate=params.dr_rate, custom_train=custom_train, custom_loss=custom_loss)
    model.summary()

    if keras_fit:
        
        train_dataset = train_dataset.repeat()
        test_dataset = test_dataset.repeat()

        model.compile(optimizer=optimizer, loss=loss_fun, run_eagerly=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

        # import ipdb; ipdb.set_trace()
        model.fit(x=x_train_mima, y=y_train_mima, batch_size=params.batch_sz, epochs=params.epochs, validation_data=(x_test_mima, y_test_mima), callbacks=[tensorboard_callback])
        # model.fit(train_dataset, steps_per_epoch=params.read_n//params.batch_sz, batch_size=params.batch_sz, \
        #    epochs=params.epochs, validation_data=test_dataset, validation_steps=params.read_n//params.batch_sz, callbacks=[tensorboard_callback])

    else:

        train_log_dir = tensorboard_log_dir + '/train'
        test_log_dir = tensorboard_log_dir + '/test'

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        # Define metrics
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
        test_accuracy = tf.keras.metrics.Mean('valid accuracy', dtype=tf.float32)

        ### train and validate ###

        for epoch in range(params.epochs):
          for (x_train, y_train) in train_dataset:
            train_step(model, optimizer, loss_fun, x_train, y_train, train_loss)
          with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
          # log weights (bias and kernel) of each layer (except input)
          log_weights(model, train_summary_writer, epoch)


          for (x_test, y_test) in test_dataset:
            test_step(model, loss_fun, x_test, y_test, valid_loss)
          with test_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=epoch)

          template = 'Epoch {}, Train Loss: {:.5f}, Test Loss: {:.5f}'
          print(template.format(epoch+1, train_loss.result(), valid_loss.result()))
          
          # Reset metrics every epoch
          train_loss.reset_states()
          valid_loss.reset_states()

    # save qr
    model_str = stco.make_qr_model_str(run_n_qr=params.qr_run_n, run_n_vae=params.vae_run_n, quantile=quantile, sig_id=params.sig_sample_id, sig_xsec=sig_xsec, strategy_id=params.strategy_id)
    model.save(os.path.join(qr_model_dir, model_str))

    # print image to tensorboard
    img_log_dir = tensorboard_log_dir + '/plots'
    os.system('rm -rf '+img_log_dir+'/*')
    img_file_writer = tf.summary.create_file_writer(img_log_dir)
    img = plot_discriminator_cut(model, qcd_test_sample, score_strategy, norm_x_mima, norm_y_mima)
    with img_file_writer.as_default():
        tf.summary.image("Training data and cut", img, step=0)

    if not keras_fit:
        train_summary_writer.flush(); train_summary_writer.close()
        test_summary_writer.flush(); test_summary_writer.close()
    img_file_writer.flush(); img_file_writer.close()
