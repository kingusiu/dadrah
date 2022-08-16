import tensorflow as tf


def quantile_accuracy_wrap(inputs, targets, predictions, bin_idx, part): 

    def quantile_accuracy(inputs, targets, predictions, bin_idx, part):
        # one accuracy metric for each bin (keras logistics): calculate target above or below prediction

        # todo: compute divergence from quantile instead of percentage (need to pass in quantile as well) -> could be used as part of loss
        # import ipdb; ipdb.set_trace()
        # divide into 4 bins on mjj
        bins = tf.constant(np.array([1199.,2200.,3200.,4200.]).astype('float32')) # instead of squeezing and reshaping expand dims of bins to (4,1)?
        # bins_t = tf.reshape(bins, shape=(tf.shape(inputs)[0],-1)) # add batch dim
        bin_idcs = tf.searchsorted(bins,tf.squeeze(inputs)) # squeeze to 1-dim for searchsort and bins
        bin_idcs = tf.reshape(bin_idcs, shape=(tf.shape(inputs)[0],1)) # reshape to batch size

        # reduces to one value per batch (how many samples below/above cut in this batch)
        if part == 'above':
            count = tf.math.count_nonzero(targets[bin_idcs==bin_idx] > predictions[bin_idcs==bin_idx])
        else:
            count = tf.math.count_nonzero(targets[bin_idcs==bin_idx] <= predictions[bin_idcs==bin_idx]) # check for security also below
        # return share of total
        return tf.math.divide(tf.cast(count,tf.float32),tf.cast(tf.size(inputs),tf.float32))

    return quantile_accuracy



class Custom_Train_Step_Model(tf.keras.Model):

    def __init__(self, **kwargs):

        super(Custom_Train_Step_Model, self).__init__(**kwargs)
        
        inputs_mjj, targets = kwargs['inputs']
        outputs = kwargs['outputs']

        # add custom metrics
        self.custom_metrics ={}
        # do this for 4 bins, passing in bin idx to quantile accuracy, aggregation = mean of batches in each epoch
        for bin_idx in range(1,5):
            for part in ['above', 'below']:
                name = name='quantile_acc_bin'+str(bin_idx)+'_'+part
                self.custom_metrics[name] = quantile_accuracy_wrap(inputs_mjj,targets,outputs,bin_idx=bin_idx,part=part)

    
    def train_step(self, data):
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self([inputs,targets], training=True)
            loss = self.compiled_loss(targets, predictions)

        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        import ipdb; ipdb.set_trace()

        for met in self.metrics:
            if met.name == 'loss':
                continue
            met(inputs, targets, predictions)

        return {'loss':loss, **{met.name: met.result() for met in self.metrics if met.name != 'loss'}}


    def test_step(self, data):

        inputs, targets = data
        predictions = self([inputs,targets], training=False)
        loss = self.compiled_loss(targets, predictions)

        return {'val_loss': loss}


    def compute_metrics(self, x, y, y_pred, sample_weight):
        # This super call updates `self.compiled_metrics` and returns results
        # for all metrics listed in `self.metrics`.
        metric_results = super(MyModel, self).compute_metrics(
            x, y, y_pred, sample_weight)
        # update custom metrics
        for metric in self.custom_metrics.values():
            metric(x,y,y_pred)
        # Note that `self.custom_metric` is not listed in `self.metrics`.
        self.custom_metric.update_state(x, y, y_pred, sample_weight)
        metric_results['custom_metric_name'] = self.custom_metric.result()
        return metric_results
