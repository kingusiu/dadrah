import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import mplhep as hep
import os

import dadrah.util.run_paths as runpa
import dadrah.selection.selection_util as seut
import dadrah.selection.loss_strategy as lost
import dadrah.util.string_constants as strc
import pofah.jet_sample as js


def plot_ortho_quantiles(sample, quantile, loss_strategy, x_range, y_range, plot_name='hist2d_ortho_quant', fig_dir='fig', fig_format='.png', cut_xmax=False):

    feature_key = 'mJJ'
    
    # Load CMS style sheet
    plt.style.use(hep.style.CMS)

    fig = plt.figure()
    x_min = np.min(sample[feature_key])
    if cut_xmax:
        x_max = np.percentile(sample[feature_key], 1e2*(1-1e-4))
    else:
        x_max = np.max(sample[feature_key])

    loss = loss_strategy(sample)

    plt.hist2d(sample[feature_key], loss, range=(x_range, y_range), \
                norm=LogNorm(), bins=200, cmap=cm.get_cmap('Blues'), cmin=0.001)

    plt.ylabel('min(L1,L2)')
    plt.xlabel('$M_{jj}$ [GeV]')
    #plt.title('quantile cuts' + title_suffix)
    plt.colorbar()
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format), bbox_inches='tight')
    plt.close(fig)




if __name__ == '__main__':

    ae_run_n = 113
    qr_run_n = 4
    sample_ids = ['qcdSigAllTestReco', 'GtoWW35naReco']
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # quantiles = [0.9]
    sig_xsec = 100.
    loss_strategy = lost.loss_strategy_dict['rk5_05']
    path_ext_dict = {'vae_run': str(ae_run_n), 'qr_run': str(qr_run_n), 'sig': sample_ids[1], 'xsec': str(int(sig_xsec)), 'loss': 'rk5_05'}

    ### paths ###

    # data inputs: /eos/user/k/kiwoznia/data/QR_results/events/vae_run_113/qr_run_2/sig_GtoWW35naReco/xsec_100/loss_rk5_05
    # data outputs (figures): fig/vae_run_113/qr_run_2/sig_GtoWW35naReco/xsec_100/loss_rk5_05/ortho_quantiles

    paths = runpa.RunPaths(in_data_dir=strc.dir_path_dict['base_dir_qr_selections'], in_data_names=strc.file_name_path_dict, out_data_dir=strc.dir_path_dict['base_dir_qr_analysis'])
    paths.extend_in_path_data(path_ext_dict)
    paths.extend_out_path_data({**path_ext_dict, 'ortho_quantiles': None})

    print('plotting figures to ' + paths.out_data_dir)

    # read data
    sample_qcd = js.JetSample.from_input_file(sample_ids[0], paths.in_file_path(sample_ids[0]))
    sample_sig = js.JetSample.from_input_file(sample_ids[1], paths.in_file_path(sample_ids[1]))

    # get shared xlimits
    xmin, xmax = min(min(sample_qcd['mJJ']), min(sample_sig['mJJ'])), max(max(sample_qcd['mJJ']), max(sample_sig['mJJ']))
    # get shared ylimits
    ymin, ymax = min(min(loss_strategy(sample_qcd)), min(loss_strategy(sample_sig))), max(max(loss_strategy(sample_qcd)), max(loss_strategy(sample_sig))) 


    for sample_id, sample in zip(sample_ids, [sample_qcd, sample_sig]):

        samples_ortho_quantiles = seut.divide_sample_into_orthogonal_quantiles(sample, quantiles)

        for sample_ortho, quantile in zip(samples_ortho_quantiles, quantiles):
            plot_ortho_quantiles(sample_ortho, quantile, loss_strategy=loss_strategy, x_range=(xmin,xmax), y_range=(ymin,ymax), plot_name='ortho_quant_'+sample_id+'_q{:02}'.format(int(quantile*100)), fig_dir=paths.out_data_dir)
