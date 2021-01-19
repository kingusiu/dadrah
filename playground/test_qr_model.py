import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import vande.vae.layers as layers


def combine_loss_min(x):
    """ L1 & L2 > LT """
    return np.minimum(x['j1TotalLoss'],x['j2TotalLoss'])

def plot_cut(model, sample, plot_name):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample['mJJ'])*0.8
    x_max = np.percentile(sample['mJJ'], 99.99)
    loss = combine_loss_min(sample)
    plt.hist2d(sample['mJJ'], loss,
           range=((x_min , x_max), (np.min(loss), np.percentile(loss, 1e2*(1-1e-4)))), 
           norm=LogNorm(), bins=100, label='signal data')

    xs = np.arange(x_min, x_max, 0.001*(x_max-x_min))
    ys = model(xs, training=False)
    # print('prediction for \n', xs, ': \n', ys.numpy().flatten())
    plt.plot(xs, ys , '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 & L2 > LT')
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.title(str(sample) + ' cut' )
    plt.colorbar()
    plt.legend(loc='best')
    plt.draw()
    fig.savefig(os.path.join('fig', plot_name + '.png'), bbox_inches='tight')
    plt.close(fig)

def normalize_data(x):
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    return x


class Custom_Train_Step_Model(tf.keras.Model):
    
    def train_step(self, data):
        inputs, targets = data
        if debug:
            import ipdb; ipdb.set_trace()

        with tf.GradientTape() as tape:
            predictions = self(inputs)
            loss = self.compiled_loss(targets, predictions)

        if debug:
            kernels = [w.numpy() for w in self.trainable_weights if len(w.shape) > 1]
            kernels_mu = list(map(np.mean, kernels))
            kernels_std = list(map(np.std, kernels))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}


def make_model(n_layers, n_nodes, x_mu_std=(0.,1.), initializer='glorot_uniform', activation='relu'):
    inputs = tf.keras.Input(shape=(1,))
    x = layers.StdNormalization(*x_mu_std)(inputs)
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(n_nodes, kernel_initializer=initializer, activation=activation)(x)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
    model = Custom_Train_Step_Model(inputs, outputs)
    return model


def quantile_loss(quantile):
    def loss(target, pred):
        err = target - pred
        return tf.where(err>=0, quantile*err, (quantile-1)*err)
    return loss


def custom_train(model, loss_fn, optimizer, x_train, y_train, batch_sz, epochs):
     train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_sz)

     for epoch in range(epochs):
        loss_per_epoch = 0
        
        for step, (x_batch, y_batch) in enumerate(train_ds):
            if debug:
                import ipdb; ipdb.set_trace()
            
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = tf.math.reduce_mean(loss_fn(y_batch, predictions))
                loss_per_epoch += loss
            
            if debug:
                kernels = [w.numpy() for w in model.trainable_weights if len(w.shape) > 1]
                kernels_mu = list(map(np.mean, kernels))
                kernels_std = list(map(np.std, kernels))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_per_epoch /= (step + 1)
        print('### [Epoch {}]: train loss {:.3f} ###'.format(epoch, loss_per_epoch))



# ************************ #
#        get data          #
# ************************ #
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_106'})
qcd_sig_sample = js.JetSample.from_input_dir('qcdSigReco', paths.sample_dir_path('qcdSigReco'), read_n=int(2e6))
print('training on {} events'.format(len(qcd_sig_sample)))
x_train = qcd_sig_sample['mJJ']
y_train = combine_loss_min(qcd_sig_sample)


# ************************ #
#        train model       #
# ************************ #
debug = False
quantile = 0.5
loss_fn = quantile_loss(quantile)
initializer = 'he_uniform'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
fit_type = 'custom' # 'keras'
activation = 'elu'

# *** model parameters *** #
if debug:
    batch_sz = 3
    n_layers = 3
    n_nodes = 5
    epochs = 10
else:
    batch_sz = 256
    n_layers = 7
    n_nodes = 70
    epochs = 25

# *** build model *** #
model = make_model(n_layers=n_layers, n_nodes=n_nodes, x_mu_std=(np.mean(x_train), np.std(x_train)), initializer=initializer, activation=activation)
model.summary()

# *** fit model *** #
if fit_type == 'keras':
    model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly=True) # Adam(lr=1e-3) TODO: add learning rate
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_sz, shuffle=True, verbose=2)
else:
    custom_train(model=model, loss_fn=loss_fn, optimizer=optimizer, x_train=x_train, y_train=y_train, batch_sz=batch_sz, epochs=epochs)

# *** predict and plot results *** #
plot_cut(model, qcd_sig_sample, plot_name='qr_cut_'+fit_type+'_fit_'+activation)
