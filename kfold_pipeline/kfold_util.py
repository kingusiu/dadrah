import os
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pofah.jet_sample as jesa


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
