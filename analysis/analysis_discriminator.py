import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os


def analyze_discriminator_cut(discriminator, sample, feature_key='mJJ', plot_name='discr_cut', fig_dir=None):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])*0.8
    x_max = np.percentile(sample[feature_key], 99.99)
    loss = discriminator.loss_strategy(sample)
    plt.hist2d(sample[feature_key], loss,
           range=((x_min , x_max), (np.min(loss), np.percentile(loss, 1e2*(1-1e-4)))), 
           norm=LogNorm(), bins=100, label='signal data')

    xs = np.arange(x_min, x_max, 0.001*(x_max-x_min))
    plt.plot(xs, discriminator.predict(xs) , '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 & L2 > LT')
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.title(str(sample) + ' ' + str(discriminator) )
    plt.colorbar()
    plt.legend(loc='best')
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'), bbox_inches='tight')
    plt.close(fig)
