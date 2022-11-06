import tensorflow as tf
import vande.vae.layers as layers


# ******************************************** #
#           quantile regression models         #
# ******************************************** #


# state of the art model set up for 2nd finite difference smoothness metric


class QrModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):

        self.regularizer = kwargs.pop('regularizer', None)
        super().__init__(*args, **kwargs)


    def compile(self, loss, metric_fn, optimizer, run_eagerly=True, **kwargs):
        (super().compile)(optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)
        self.quant_loss_fn = loss
        self.metric_fn = metric_fn
        self.loss_mean = tf.keras.metrics.Mean('loss')
        self.metric_mean = tf.keras.metrics.Mean(self.metric_fn.name)
        self.reg_mean = tf.keras.metrics.Mean('reg')

    @tf.function
    def train_step(self, data):

        inputs, targets = data
        
        with tf.GradientTape() as (tape):
            predictions = self([inputs, targets], training=True)
            loss = self.quant_loss_fn(targets, predictions)
            reg_loss = tf.add_n(model.losses) # add regularization loss
            total_loss = loss + reg_loss
        
        trainable_variables = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        delta = self.metric_fn.delta
        # import ipdb; ipdb.set_trace()
        pred = self([inputs, targets], training=False) # compute new predictions after backpropagation step
        pred_delta_plus = self([inputs+delta, targets], training=False)
        pred_delta_minus = self([inputs-delta, targets], training=False)
        norm_delta = tf.math.divide_no_nan(delta,self.get_layer('Normalization').std_x) # scale delta like inputs
        metric_val = self.metric_fn(pred, pred_delta_plus, pred_delta_minus, norm_delta)

        self.loss_mean.update_state(loss)
        self.metric_mean.update_state(metric_val)
        self.reg_mean.update_state(reg_loss)

        return {'loss':self.loss_mean.result(), self.metric_fn.name : self.metric_mean.result(), 'reg' : self.reg_mean.result()}


    def test_step(self, data):

        inputs, targets = data

        predictions = self([inputs, targets], training=False)

        loss = self.quant_loss_fn(targets, predictions)

        delta = self.metric_fn.delta
        # import ipdb; ipdb.set_trace()
        pred_delta_plus = self([inputs+delta, targets], training=False)
        pred_delta_minus = self([inputs-delta, targets], training=False)
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


    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)




### custom train and test step model with optional quantile-ratio-deviation loss term

class QrModelRatios(tf.keras.Model):

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
                reg_loss = tf.math.add_n(self.losses)
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





class QuantileRegression():

    def __init__(self, quantile, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu'):
        self.quantile = quantile
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.initializer = initializer
        self.activation = activation

    def build(self):
        inputs = tf.keras.Input(shape=(1,))
        x = layers.StdNormalization(*self.x_mu_std)(inputs)
        for _ in range(self.n_layers):
            x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)
        outputs = tf.keras.layers.Dense(1, kernel_initializer=self.initializer)(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model


class QuantileRegressionPoly():

    def __init__(self, quantile, x_mu_std=(0.,1.), optimizer='adam',  activation='linear'):
        self.quantile = quantile
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.activation = activation

    def build(self):

        inputs = tf.keras.Input(shape=(1,))
        normx = layers.StdNormalization(*self.x_mu_std)(inputs)

        x_cubed = tf.keras.layers.Lambda(lambda x:x**3)(normx)
        x_squared = tf.keras.layers.Lambda(lambda x:x**2)(normx)
        hidden = tf.keras.layers.Concatenate()([x_cubed,x_squared,normx])
        outputs = tf.keras.layers.Dense(1, activation = self.activation)(hidden)

        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer)

        model.summary()
        return model


class QuantileRegressionBernstein():

    def __init__(self, quantile, x_mu_std=(0.,1.), optimizer='adam',  activation='linear'):
        self.quantile = quantile
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.activation = activation

    def build(self):

        inputs = tf.keras.Input(shape=(1,))
        normx = layers.StdNormalization(*self.x_mu_std)(inputs)

        b03 = tf.keras.layers.Lambda(lambda x: 1 - 3*x + 3*x**2 - x**3)(normx)
        b13 = tf.keras.layers.Lambda(lambda x: 3*x - 6*x**2 + 3*x**3)(normx)
        b23 = tf.keras.layers.Lambda(lambda x: 3*x**2 - 3*x**3)(normx)
        b33 = tf.keras.layers.Lambda(lambda x: x**3)(normx)

        hidden = tf.keras.layers.Concatenate()([b03, b13, b23, b33])
        outputs = tf.keras.layers.Dense(1, activation = self.activation)(hidden)

        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer)

        model.summary()
        return model
