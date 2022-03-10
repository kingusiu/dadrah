import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.analysis.root_plotting_util as rpu
import dadrah.util.run_paths as runpa
import pofah.util.sample_factory as sf
import pofah.jet_sample as js
import pofah.util.utility_fun as utfu
import pofah.util.experiment as exp

import pathlib
import argparse

''' 
    need to source environment that has access to ROOT before launching this script!
    e.g. source /cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-centos7-gcc9-opt/setup.sh
'''


def plot_mjj_spectrum(sample, quantile, fig_dir='fig'):
    inv_quant = round((1.-quantile),2)
    title = sample.name + ": BG like vs SIG like mjj distribution and ratio qnt {}".format(int(quantile*100))
    plot_name = 'mJJ_ratio_bg_vs_sig_' + sample.name + '_q' + str(int(quantile*100))
    print('plotting {} to {}'.format(plot_name, fig_dir))
    # selections saved as inverse quantiles because of code in dijet fit (i.e. if quantile = 0.90, s.t. 90% of points are BELOW threshold, this is saved as 0.1 or q10, meaning that 10% of points are ABOVE threshold = inv_quant)
    rpu.make_bg_vs_sig_ratio_plot(sample.rejected(inv_quant, mjj_key), sample.accepted(inv_quant, mjj_key), target_value=inv_quant, n_bins=60, title=title, plot_name=plot_name, fig_dir=fig_dir)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='run mjj spectrum analysis with QR cuts applied')
    parser.add_argument('-x', dest='sig_xsec', type=float, default=100., help='signal injection cross section')
    args = parser.parse_args()

    run = 113
    sample_ids = ['qcdSigAllTestReco', 'GtoWW35brReco']
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # quantiles = [0.9]
    mjj_key = 'mJJ'
    param_dict = {'$run$': str(run), '$sig_name$': sample_ids[1], '$sig_xsec$': str(int(args.sig_xsec))}

    input_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path(param_dict) # in selection paths new format with run_x, sig_x, ...
    fig_dir = exp.Experiment(run_n=run, param_dict=param_dict).setup(analysis_dir_qr=True).analysis_dir_qr_mjj

    for sample_id in sample_ids:
        for quantile in quantiles:
            sample = js.JetSample.from_input_file(sample_id, input_paths.sample_file_path(sample_id))
            plot_mjj_spectrum(sample, quantile, fig_dir)
