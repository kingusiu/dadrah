import numpy as np
import time
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import sklearn.ensemble as scikit
import dadrah.selection.quantile_regression as qr
import pofah.jet_sample as js
import vande.training as train
import vande.vae.layers as layers


class Discriminator(metaclass=ABCMeta):

    def __init__(self, quantile, loss_strategy):
        self.loss_strategy = loss_strategy
        self.quantile = quantile
        self.mjj_key = 'mJJ'

    @abstractmethod
    def fit(self, jet_sample):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass 

    @abstractmethod
    def predict(self, data):
        '''predict cut for each example in data'''
        pass

    @abstractmethod
    def select(self, jet_sample):
        pass

    def __repr__(self):
        return '{}% qnt, {} strategy'.format(str(self.quantile*100), self.loss_strategy.title_str)


class FlatCutDiscriminator(Discriminator):

    def fit(self, jet_sample):
        loss = self.loss_strategy(jet_sample)
        self.cut = np.percentile( loss, (1.-self.quantile)*100 )
        
    def predict(self, jet_sample):
        return np.asarray([self.cut]*len(jet_sample))

    def select(self, jet_sample):
        loss = self.loss_strategy(jet_sample)
        return loss > self.cut

    def __repr__(self):
        return 'Flat Cut: ' + Discriminator.__repr__(self)


class QRDiscriminator(Discriminator):

    def __init__(self, quantile, loss_strategy, batch_sz=128, epochs=100, learning_rate=0.001, optimizer=tf.keras.optimizers.Adam, **model_params):
        Discriminator.__init__(self, quantile, loss_strategy)
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer() #optimizer(learning_rate)
        self.model_params = model_params

    ''' not properly implemented! '''
    @classmethod
    def from_saved_model(model_path):
        cls = type(self)
        # need to read quantile and loss strategy here (like beta in vae)
        instance = cls(quantile=0.5, loss_strategy=None) # dummy init values
        instance.load(model_path)
        return instance

    @tf.function
    def training_step(self, x_batch, y_batch):
        # Open a GradientTape to record the operations run in forward pass
        # import ipdb; ipdb.set_trace()
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss_value = tf.math.reduce_mean(qr.quantile_loss(y_batch, predictions, self.quantile))

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def training_epoch(self, train_dataset):
        # Iterate over the batches of the dataset.
        train_loss = 0.
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = self.training_step(x_batch, y_batch)
            train_loss += loss_value
            if step % 10000 == 0:
                print("Step {}: lr {:.3e}, loss {:.4f}".format(step, self.optimizer.learning_rate.numpy(), loss_value))
        return train_loss / (step + 1)

    def valid_epoch(self, valid_dataset):
        # Iterate over the batches of the dataset.
        valid_loss = 0.
        for step, (x_batch, y_batch) in enumerate(valid_dataset):
            predictions = self.model(x_batch, training=False)
            valid_loss += tf.math.reduce_mean(qr.quantile_loss(y_batch, predictions, self.quantile))
        return valid_loss / (step + 1)


    def make_training_datasets(self, train_sample, valid_sample):
        x_train = train_sample[self.mjj_key]
        y_train = self.loss_strategy(train_sample)
        x_valid = valid_sample[self.mjj_key]
        y_valid = self.loss_strategy(valid_sample)
        return (x_train, y_train), (x_valid, y_valid)

    def fit(self, train_sample, valid_sample):
        # process the input
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_sz)#.shuffle(self.batch_sz*10)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(self.batch_sz)

        # build the regressor
        self.regressor = qr.QuantileRegressionV2(**self.model_params)
        # self.model = self.regressor.make_model(x_mean_std=(np.mean(x_train), np.std(x_train)), y_mean_std=(np.mean(y_train), np.std(y_train)))
        self.model = self.regressor.make_model(x_min_max=(np.min(x_train), np.max(x_train)), y_min_max=(np.min(y_train), np.max(y_train)))
        
        # build loss arrays and callbacks
        losses_train = []
        losses_valid = []
        train_stop = train.Stopper(optimizer=self.optimizer)

        # run training
        for epoch in range(self.epochs):
            start_time = time.time()
            losses_train.append(self.training_epoch(train_dataset))
            losses_valid.append(self.valid_epoch(valid_dataset))
            # print epoch results
            print('### [Epoch {} - {:.2f} sec]: train loss {:.3f}, val loss {:.3f} (mean / batch) ###'.format(epoch, time.time()-start_time, losses_train[-1], losses_valid[-1]))
            if train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break

        return losses_train, losses_valid


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'StdNormalization': layers.StdNormalization, 'StdUnnormalization': layers.StdUnnormalization}, compile=False)
        print('loaded model ', self.model)
        return self

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        return self.model(data, training=False)

    def select(self, jet_sample):
        loss_cut = self.predict(jet_sample)
        return self.loss_strategy(jet_sample) > loss_cut

    def __repr__(self):
        return 'QR Cut: ' + Discriminator.__repr__(self)


class QRDiscriminator_KerasAPI(QRDiscriminator):
    """docstring for QRDiscriminator_KerasAPI"""
    def __init__(self, **kwargs):
        super(QRDiscriminator_KerasAPI, self).__init__(**kwargs)
        self.model_class = qr.QuantileRegression

    def fit(self, train_sample, valid_sample):
        # prepare training set
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)

        self.model = self.model_class(quantile=self.quantile, x_mu_std=(np.mean(x_train), np.std(x_train)), **self.model_params).build()
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_sz, verbose=2, validation_data=(x_valid, y_valid), shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])
       
        return self.history.history['loss'], self.history.history['val_loss']

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        xx = data #self.scale_input(data)
        predicted = self.model.predict(xx).flatten() 
        # return self.unscale_output(predicted)
        return predicted


class QRDiscriminatorPoly_KerasAPI(QRDiscriminator_KerasAPI):

    """docstring for QRDiscriminator_KerasAPI"""

    def __init__(self, **kwargs):
        super(QRDiscriminator_KerasAPI, self).__init__(**kwargs)
        self.model_class = qr.QuantileRegressionPoly
        

class QRDiscriminatorBernstein_KerasAPI(QRDiscriminator_KerasAPI):

    """docstring for QRDiscriminator_KerasAPI"""

    def __init__(self, **kwargs):
        super(QRDiscriminator_KerasAPI, self).__init__(**kwargs)
        self.model_class = qr.QuantileRegressionBernstein
        

class GBRDiscriminator(Discriminator):

    def fit(self, jet_sample):
        self.model = scikit.GradientBoostingRegressor(loss='quantile', alpha=1-self.quantile, learning_rate=.01, max_depth=2, verbose=2)
