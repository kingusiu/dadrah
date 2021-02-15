import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.analysis.root_plotting_util as rpu
import pofah.util.sample_factory as sf
import pofah.jet_sample as js
import pofah.util.utility_fun as utfu
import dadrah.util.path_constants as paco

import pathlib

''' 
    need to source environment that has access to ROOT before launching this script!
    e.g. source /cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-centos7-gcc9-opt/setup.sh
'''


def plot_mjj_spectrum(sample, quantile, fig_dir='fig'):
    inv_quant = round((1.-quantile),2)
    title = sample.name + ": BG like vs SIG like mjj distribution and ratio qnt {}".format(int(quantile*100))
    plot_name = 'mJJ_ratio_bg_vs_sig_' + sample.name + '_q' + str(int(quantile*100))
    print('plotting {} to {}'.format(plot_name, fig_dir))
    rpu.make_bg_vs_sig_ratio_plot(sample.rejected(inv_quant, mjj_key), sample.accepted(inv_quant, mjj_key), target_value=inv_quant, n_bins=60, title=title, plot_name=plot_name, fig_dir=fig_dir)


if __name__ == '__main__':
    
    run = 113
    sample_ids = ['qcdSigAllTestReco', 'GtoWW35naReco']
    sig_xsec = 100.
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # quantiles = [0.9]
    mjj_key = 'mJJ'

    input_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(run), '$sig_name$': sample_ids[1], '$sig_xsec$': str(int(sig_xsec))}) # in selection paths new format with run_x, sig_x, ...
    fig_dir = utfu.multi_replace(paco.analysis_paths['mJJ_spectra'], {'$run$' : str(run)})
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)            


    for sample_id in sample_ids:
        for quantile in quantiles:
            sample = js.JetSample.from_input_file(sample_id, input_paths.sample_file_path(sample_id))
            plot_mjj_spectrum(sample, quantile, fig_dir)
