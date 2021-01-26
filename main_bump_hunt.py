import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from collections import namedtuple
import datetime

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import analysis.analysis_discriminator as andi
import vande.training as train

def make_qr_model_str(run, quantile, date=True):
    date_str = ''
    if date:
        date = datetime.date.today()
        date_str = '_{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    return 'QRmodel_run_{}_qnt_{}_{}.h5'.format(run, str(int(quantile*100)), date_str)
    


#****************************************#
#           set runtime params
#****************************************#
# for a fixed signal G_RS na 3.5TeV
run = 106
sig_sample_id = 'GtoWW35naReco'
# for a fixed xsec 10
xsec = 10.
sig_in_training_num = 150
quantiles = [0.1, 0.5, 0.9]
result_dir = os.path.join('/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_results/', 'run_'+str(run))

Parameters = namedtuple('Parameters','run_n, qcd_sample_id, strategy')
params = Parameters(run_n=109, qcd_sample_id='qcdSigReco', strategy=lost.loss_strategy_dict['s5'])

#****************************************#
#           read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})

### read in mixed training sample
# qcd
qcd_sr_sample = js.JetSample.from_input_dir(params.qcd_sample_id, paths.sample_dir_path(params.qcd_sample_id), read_n=int(1e6)) # can train on max 4M events (20% of qcd SR)
qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sr_sample, 0.2)
# signal
sig_sample = js.JetSample.from_input_dir(sig_sample_id, paths.sample_dir_path(sig_sample_id))
sig_train = sig_sample.sample(n=sig_in_training_num)
# merge qcd and signal
mixed_sample = qcd_train.merge(sig_train)
# split training data into train and validation set
mixed_sample_train, mixed_sample_valid = js.split_jet_sample_train_test(mixed_sample, 0.8)
print('training on {} events, validating on {}'.format(len(mixed_sample_train), len(mixed_sample_valid)))


# ********************************************
#               train and predict
# ********************************************

do_bump_hunt = False

model_paths = []
qcd_sel_paths = []
sig_sel_paths = []
# for quantiles 0.1 0.5 0.9
for quantile in quantiles:
    
    # train QR on x_train and y_train and quantile q
    print('\ntraining QR for quantile {}'.format(quantile))    
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=params.strategy, batch_sz=256, epochs=70,  n_layers=5, n_nodes=50)
    losses_train, losses_valid = discriminator.fit(mixed_sample_train, mixed_sample_valid)

    # save the model   
    model_str = make_qr_model_str(params.run_n, quantile)
    train.plot_training_results(losses_train, losses_valid, plot_suffix=model_str[:-3], fig_dir='fig')
    model_paths.append('models/{}'.format(model_str))
    discriminator.save(model_paths[-1])
    print('saving model ', model_str)

    # prediction
    print('running prediction and writing selections to ', result_dir)

    # predict qcd_test
    selection = discriminator.select(qcd_test)
    qcd_test.add_feature('sel', selection)
    qcd_test.dump(os.path.join(result_dir, 'qcd_sr_sel_q'+str(int(quantile*100))+'.h5'))

    # predict signal test
    selection = discriminator.select(sig_sample)
    sig_sample.add_feature('sel', selection)
    sig_sample.dump(os.path.join(result_dir, sig_sample_id+'_sel_q'+str(int(quantile*100))+'.h5'))

# plot results
discriminator_list = []
for q, model_path in zip(quantiles,model_paths):
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=q, loss_strategy=params.strategy)
    discriminator.load(model_path)
    discriminator_list.append(discriminator)

andi.analyze_multi_quantile_discriminator_cut(discriminator_list, mixed_sample_valid, fig_dir='fig')

# do bump hunt
if do_bump_hunt:
    os.system("python DiJetFit.py -bg {} -sig {}".format(qcd_sel_paths, sig_sel_paths))  
