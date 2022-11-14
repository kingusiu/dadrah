import os
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from collections import defaultdict
import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"]})

import pofah.jet_sample as jesa
import dadrah.util.string_constants as stco
import dadrah.kfold_pipeline.kfold_string_constants as kstco


def read_kfold_datasets(path, kfold_n, read_n=None): # -> list(jesa.JetSample)
    file_names = ['qcd_sqrtshatTeV_13TeV_PU40_NEW_fold'+str(k+1)+'.h5' for k in range(kfold_n)]
    print('reading ' + ' '.join(file_names) + 'from ' + path)
    return [jesa.JetSample.from_input_file('qcd_fold'+str(k+1),os.path.join(path,ff),read_n=read_n) for k,ff in enumerate(file_names)]


def plot_discriminator_cut(discriminator, sample, score_strategy, feature_key='mJJ', plot_name='discr_cut', fig_dir=None, plot_suffix='', xlim=False):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])
    x_max = np.max(sample[feature_key])
    an_score = score_strategy(sample)
    x_top = np.percentile(sample[feature_key], 99.999) if xlim else x_max
    x_range = ((x_min * 0.9,x_top), (np.min(an_score), np.percentile(an_score, 99.99)))
    plt.hist2d((sample[feature_key]), an_score, range=x_range, norm=(LogNorm()), bins=100, cmap=cm.get_cmap('Blues'), cmin=0.001)
    xs = np.arange(x_min, x_max, 0.001 * (x_max - x_min))
    plt.plot(xs, (discriminator.predict([xs, xs])), '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 \& L2 > LT')
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



# *************************************************************************#
#                           paths


def get_model_paths(params, qr_model_dir):

    model_paths = defaultdict(dict)

    for quantile in params.quantiles:

        q_str = 'q'+str(int(quantile*100))

        for k in range(1,params.kfold_n+1):

            # qr full file path
            model_dir = kstco.get_qr_model_dir(params)
            model_str = kstco.get_qr_model_file_name(params,quantile,k)
            model_paths[q_str]['fold' + str(k)] = os.path.join(qr_model_dir, model_str)

    return model_paths


# *************************************************************************#
#                           binnings

def get_dijet_bins(start=0,bin_centers=False):

    bins = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                    2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                    4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')
    if bin_centers:
        bins = [(high+low)/2. for low, high in zip(bins[:-1], bins[1:])]

    return np.array(bins[start:])


def get_bins_from_envelope(params):

    bin_idx = 0
    envelope_json_path = kstco.get_envelope_full_path(params,k=1) # load envelope path

    ff = open(envelope_json_path)
    env = json.load(ff)
        
    return np.asarray(env[str(params.quantiles[0])])[:,bin_idx]