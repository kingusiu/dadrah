import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex


# read in qcd signal region sample
run_n = 101
qcd_sig_id = 'qcdSigReco'

experiment = ex.Experiment(run_n)
data_img_vae = sf.read_results_to_jet_sample_dict([qcd_sig_id], experiment, mode='img-local')


# split into discriminator train and test-set
qcd_sig_train, qcd_sig_test = ...


# train selector (and plot results => TODO)
selector = ...
selector.train(qcd_sig_train)

# apply selection to datasample
qcd_sig_train_selected = selector.apply(qcd_sig_train)
qcd_sig_test_selected = selector.apply(qcd_sig_test)

# plot mjj qcd sig to check for flat ratio


# plot mjj accepted vs rejected signal to check for mass sculpting

# do statistical analysis