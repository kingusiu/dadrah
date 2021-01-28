import os
import subprocess
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

def make_qr_model_str(run, quantile, sig_xsec, date=True):
    date_str = ''
    if date:
        date = datetime.date.today()
        date_str = '_{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    return 'QRmodel_run_{}_qnt_{}_sigx_{}_{}.h5'.format(run, str(int(quantile*100)), sig_xsec, date_str)
    

#****************************************#
#           set runtime params
#****************************************#
# for a fixed signal G_RS na 3.5TeV
run = 106
sig_sample_id = 'GtoWW35naReco'
# for a fixed xsec 10
xsecs = [10., 1.]
sig_in_training_nums = [15, 2]
quantiles = [0.1, 0.5, 0.9, 0.99]
result_dir = os.path.join('/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_results/', 'run_'+str(run))

Parameters = namedtuple('Parameters','run_n, qcd_sample_id, strategy')
params = Parameters(run_n=run, qcd_sample_id='qcdSigReco', strategy=lost.loss_strategy_dict['s5'])

do_bump_hunt = True
train_qr = True

#****************************************#
#           read in data
#****************************************#
experiment = ex.Experiment(params.run_n)
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})

# for each true signal cross-section
for xsec, sig_in_training_num in zip(xsecs, sig_in_training_nums):

    if train_qr:
        ### read in mixed training sample
        # qcd
        qcd_sr_sample = js.JetSample.from_input_dir(params.qcd_sample_id, paths.sample_dir_path(params.qcd_sample_id), read_n=int(1e6)) 
        qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sr_sample, 0.2) # can train on max 4M events (20% of qcd SR)
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


    if train_qr:

        model_paths = []
        # for quantiles 0.1 0.5 0.9, 0.99
        for quantile in quantiles:
            
            # train QR on x_train and y_train and quantile q
            print('\ntraining QR for quantile {}'.format(quantile))    
            discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=params.strategy, batch_sz=256, epochs=70,  n_layers=5, n_nodes=50)
            losses_train, losses_valid = discriminator.fit(mixed_sample_train, mixed_sample_valid)

            # save the model   
            model_str = make_qr_model_str(params.run_n, quantile, xsec)
            train.plot_training_results(losses_train, losses_valid, plot_suffix=model_str[:-3], fig_dir='fig')
            model_paths.append('models/{}'.format(model_str))
            discriminator.save(model_paths[-1])
            print('saving model ', model_str)

            # prediction
            print('running prediction')

            # predict qcd_test
            selection = discriminator.select(qcd_test)
            qcd_test.add_feature('sel_q{:02}'.format(int(round((1.-quantile),2)*100)), selection)

            # predict signal test
            selection = discriminator.select(sig_sample)
            sig_sample.add_feature('sel_q{:02}'.format(int(round((1.-quantile),2)*100)), selection)

        # write results for all quantiles
        print('writing selections to ', result_dir)
        qcd_test.dump(os.path.join(result_dir, sdfr.path_dict['file_names'][params.qcd_sample_id]+'.h5'))
        sig_sample.dump(os.path.join(result_dir, sdfr.path_dict['file_names'][sig_sample_id]+'.h5'))

        # plot results
        discriminator_list = []
        for q, model_path in zip(quantiles,model_paths):
            discriminator = disc.QRDiscriminator_KerasAPI(quantile=q, loss_strategy=params.strategy)
            discriminator.load(model_path)
            discriminator_list.append(discriminator)

        andi.analyze_multi_quantile_discriminator_cut(discriminator_list, mixed_sample_valid, plot_name='multi_discr_cut_x'+str(int(xsec)), fig_dir='fig')

    # do bump hunt
    if do_bump_hunt:
        dijet_dir = '/eos/home-k/kiwoznia/dev/vae_dijet_fit/VAEDijetFit'
        cmd = "python run_dijetfit.py --run 1 -i {} -M 3500 --sig {}.h5 --sigxsec {} --qcd {}.h5".format(result_dir, sdfr.path_dict['file_names'][sig_sample_id], xsec, sdfr.path_dict['file_names'][params.qcd_sample_id])
        print("running ", cmd)
        subprocess.check_call('pwd && source setupenv.sh && ' + cmd, cwd=dijet_dir, shell=True, executable="/bin/bash")  
        print('finished')