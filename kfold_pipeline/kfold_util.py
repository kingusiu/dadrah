import os
import io
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib import colors
from collections import defaultdict
import pathlib
import matplotlib.pyplot as plt
# plt.rcParams.update({
# "text.usetex": False,
# "font.family": "sans-serif",})
# "font.sans-serif": ["Helvetica"]

import pofah.jet_sample as jesa
import dadrah.util.string_constants as stco
import dadrah.kfold_pipeline.kfold_string_constants as kstco


def read_kfold_datasets(path, kfold_n, read_n=None): # -> list(jesa.JetSample)
    file_names = ['qcd_sqrtshatTeV_13TeV_PU40_NEW_fold'+str(k+1)+'.h5' for k in range(kfold_n)]
    print('reading ' + ' '.join(file_names) + 'from ' + path)
    return [jesa.JetSample.from_input_file('qcd_fold'+str(k+1),os.path.join(path,ff),read_n=read_n) for k,ff in enumerate(file_names)]


def plot_discriminator_cut(discriminator, sample, score_strategy, feature_key='mJJ', plot_name='discr_cut', fig_dir=None, plot_suffix='', xlim=False):

    # setup colormap for 2dhist
    cmap = cm.get_cmap('Blues')
    xc = np.linspace(0.0, 1.0, 100)
    color_list = cmap(xc)
    color_list = np.vstack((color_list[0], color_list[40:])) # keep white, drop light colors 
    my_cm = colors.ListedColormap(color_list)

    feature_key = 'mJJ'

    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])
    x_max = np.max(sample[feature_key])
    an_score = score_strategy(sample)
    x_top = np.percentile(sample[feature_key], 99.999) if xlim else x_max
    x_range = ((x_min * 0.9,x_top), (np.min(an_score), np.percentile(an_score, 99.99)))
    plt.hist2d((sample[feature_key]), an_score, range=x_range, norm=(LogNorm()), bins=100, cmap=my_cm, cmin=0.001)
    xs = np.arange(x_min, x_max, 0.001 * (x_max - x_min))
    plt.plot(xs, (discriminator.predict([xs, xs])), '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('min(L1,L2)')
    # plt.xlabel('$M_{jj}$ [GeV]')
    plt.xlabel('Mjj [GeV]')
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

def get_bins(bin_type, **kwargs):
    if bin_type == 'dijet':
        return get_dijet_bins(**kwargs)
    elif bin_type == 'expo':
        return get_expo_bins(**kwargs)
    else:
        return get_linear_bins(**kwargs)


def get_dijet_bins(bin_start=0, bin_centers=True):

    bins = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                    2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                    4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')
    if bin_centers:
        bins = [(high+low)/2. for low, high in zip(bins[:-1], bins[1:])]

    return np.array(bins[bin_start:])


def get_expo_bins(n_bins=40, min_mjj=1200., max_mjj=6000, bin_centers=True):
    ''' exponentially expanding bin-width '''
    x_shift = 3
    lin_bins = np.linspace(0.,1.,n_bins)
    exp_bins = lin_bins/(np.exp(-lin_bins+x_shift)/np.exp(x_shift-1))
    bins = exp_bins*(max_mjj-min_mjj)+min_mjj
    if bin_centers:
        bins = [(high+low)/2. for low, high in zip(bins[:-1], bins[1:])]
    return np.asarray(bins)


def get_linear_bins(n_bins=40, min_mjj=1200., max_mjj=6000, bin_centers=True):
    bins = np.array(np.linspace(min_mjj, max_mjj, n_bins).tolist()).astype('float')
    if bin_centers:
        bins = [(high+low)/2. for low, high in zip(bins[:-1], bins[1:])]
    return np.asarray(bins)


def get_bins_from_envelope(params):

    bin_idx = 0
    envelope_json_path = kstco.get_envelope_file(params,k=1) # load envelope path

    ff = open(envelope_json_path)
    env = json.load(ff)
        
    return np.asarray(env[str(params.quantiles[0])])[:,bin_idx]
