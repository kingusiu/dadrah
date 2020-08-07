import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import selection.discriminator as dis
import selection.loss_strategy as ls


# read in qcd signal region sample
run_n = 101
qcd_sig_id = 'qcdSigReco'

experiment = ex.Experiment(run_n)
qcd_sig_data = sf.read_results_to_jet_sample_dict([qcd_sig_id], experiment)



# split into discriminator train and test-set

# define quantile and loss-strategy for discimination
quantile = 0.05 # 5%
strategy = ls.combine_loss_min

print(qcd_sig_data['qcdSigReco'].features())
# train discriminator (and plot results => TODO)
discriminator = dis.FlatCutDiscriminator(quantile=quantile, loss_strategy=strategy)
discriminator.fit(qcd_sig_data['qcdSigReco'])

# apply selection to datasample
qcd_sig_train_selected = discriminator.select(qcd_sig_data['qcdSigReco'])
#qcd_sig_test_selected = discriminator.apply(qcd_sig_test)

# plot mjj qcd sig to check for flat ratio


# plot mjj accepted vs rejected signal to check for mass sculpting

# do statistical analysis