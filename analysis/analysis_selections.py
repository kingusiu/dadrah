import dadrah.analysis.root_plotting_util as rpu
import dadrah.util.run_paths as runpa
import dadrah.util.string_constants as stco
import dadrah.selection.selection_util as seut
import pofah.util.sample_factory as sf
import pofah.jet_sample as js
import pofah.util.utility_fun as utfu
import pofah.util.experiment as exp

import pathlib
import argparse
from typing import List

''' 
    need to source environment that has access to ROOT before launching this script!
    e.g. source /cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-centos7-gcc9-opt/setup.sh
'''


def plot_mjj_spectrum(sample:js.JetSample, quantile:float, mjj_key:str='mJJ', fig_dir:str='fig'):
    inv_quant = round((1.-quantile),2)
    title = sample.name + ": BG like vs SIG like mjj distribution and ratio qnt {}".format(int(quantile*100))
    plot_name = 'mJJ_ratio_bg_vs_sig_' + sample.name + '_q' + str(int(quantile*100))
    print('plotting {} to {}'.format(plot_name, fig_dir))
    # selections saved as inverse quantiles because of code in dijet fit (i.e. if quantile = 0.90, s.t. 90% of points are BELOW threshold, this is saved as 0.1 or q10, meaning that 10% of points are ABOVE threshold = inv_quant)
    # -> use 1-q in all own scripts
    rpu.make_bg_vs_sig_ratio_plot(sample.rejected(inv_quant, mjj_key), sample.accepted(inv_quant, mjj_key), target_value=inv_quant, n_bins=60, title=title, plot_name=plot_name, fig_dir=fig_dir)


def plot_mjj_spectrum_ortho_quantiles(sample:js.JetSample, quantiles:List, mjj_key:str='mJJ', fig_dir:str='fig'):

    # split dataset into orthogonal subsets by quantile cut
    samples_ortho_quantiles = seut.divide_sample_into_orthogonal_quantiles(sample, quantiles)
    # get mJJ values
    mjj_vals = [sample_ortho['mJJ'] for sample_ortho in samples_ortho_quantiles]
    # plot ratio hist
    rpu.make_quantile_template_ratio_plot(mjj_vals, quantiles, fig_dir=fig_dir)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='run mjj spectrum analysis with QR cuts applied')
    parser.add_argument('-x', dest='sig_xsec', type=float, default=0., help='signal injection cross section')
    args = parser.parse_args()

    ae_run_n = 113
    qr_run_n = 4
    sample_ids = ['qcdSigAllTestReco', 'GtoWW35naReco']
    sample_id_qcd = sample_ids[0]
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # quantiles = [0.9]
    path_ext_dict = {'vae_run': str(ae_run_n), 'qr_run': str(qr_run_n), 'sig': sample_ids[1], 'xsec': str(int(args.sig_xsec)), 'loss': 'rk5_05'}

    ### paths ###

    # data inputs: /eos/user/k/kiwoznia/data/QR_results/events/ae_run_113/qr_run_2/sig_GtoWW35naReco/xsec_100/loss_rk5_05
    # data outputs (figures): fig/vae_run_113/qr_run_2/sig_GtoWW35naReco/xsec_0/loss_rk5_05//mjj_spectra

    paths = runpa.RunPaths(in_data_dir=stco.dir_path_dict['base_dir_qr_selections'], in_data_names=stco.file_name_path_dict, out_data_dir=stco.dir_path_dict['base_dir_qr_analysis'])
    paths.extend_in_path_data(path_ext_dict)
    paths.extend_out_path_data({**path_ext_dict, 'mjj_spectra': None})

    samples = {}

    for sample_id in sample_ids:
        samples[sample_id] = js.JetSample.from_input_file(sample_id, paths.in_file_path(sample_id))
        # for quantile in quantiles:
        #     plot_mjj_spectrum(samples[sample_id], quantile, fig_dir=paths.out_data_dir)


    # plot orthogonal quantile mjj spectra

    paths.extend_out_path_data({'orthogonal': None})

    # make (hopefully) flat ratio plot for xsec 0 training and no signal injection
    plot_mjj_spectrum_ortho_quantiles(samples[sample_id_qcd], quantiles, fig_dir=paths.out_data_dir)