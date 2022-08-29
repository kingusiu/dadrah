import tensorflow as tf
import vande.vae.layers as layers


# ******************************************** #
#           quantile regression loss           #
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
                count_above = tf.math.count_nonzero(targets[bin_mask] > predictions[bin_mask])
                ratio = tf.math.divide_no_nan(tf.cast(count_above,tf.float32),tf.cast(count_tot,tf.float32))
                ratios[bin_idx-1].assign(ratio)

        return tf.reduce_sum(tf.math.square(ratios-self.quantile)) # sum over m bins 

# ******************************************** #
#           quantile regression models         #
# ******************************************** #


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
