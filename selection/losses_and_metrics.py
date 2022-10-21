import tensorflow as tf


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


class quantile_loss_smooth(tf.keras.losses.Loss):
    ''' S. Zheng, “Gradient descent algorithms for quantile regression with
    smooth approximation,” International Journal of Machine Learning and
    Cybernetics, vol. 2, no. 3, p. 191, 2011. '''

    def __init__(self, quantile, beta=0.2, name='quantileLossSmooth'):
        super().__init__(name=name)
        self.quantile = tf.constant(quantile)
        self.beta = tf.constant(beta) # smoothing constant

    @tf.function
    def call(self, targets, predictions):
        targets = tf.squeeze(targets)
        predictions = tf.squeeze(predictions)
        err = tf.subtract(targets, predictions)
        return tf.reduce_mean(self.quantile * err + self.beta * tf.math.log(1. + tf.math.exp(-tf.math.divide_no_nan(err,self.beta)))) # return mean over samples in batch



class quantile_dev_loss():

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


class binned_quantile_dev_loss():

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
