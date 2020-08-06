import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex


# read in qcd signal region sample
run_n = 101
qcd_sig_id = 'qcdSigReco'

experiment = ex.Experiment(run_n)
data_img_vae = sf.read_results_to_jet_sample_dict([qcd_sig_id], experiment, mode='img-local')

