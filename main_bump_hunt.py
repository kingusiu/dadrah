import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype
import pathlib
import copy

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.selection.discriminator as disc
import dadrah.selection.anomaly_score_strategy as ansc
import dadrah.selection.qr_workflow as qrwf
import analysis.analysis_discriminator as andi
import dadrah.util.data_processing as dapr
import dadrah.util.string_constants as stco
import pofah.phase_space.cut_constants as cuts


#****************************************#
#           set runtime params
#****************************************#

grs_na_15_x10fb = 106.12 # number of expected events of g_rs narrow at 1.5TeV
grs_na_25_x10fb = 110.06 # number of expected events of g_rs narrow at 2.5TeV
grs_na_35_x10fb = 112.27 # number of expected events of g_rs narrow at 3.5TeV
grs_na_45_x10fb = 113.98 # number of expected events of g_rs narrow at 4.5TeV

signal_contamin = { 'GtoWW35naReco' : { 0: 0,
                                        10: int(grs_na_35_x10fb),
                                        20: int(grs_na_35_x10fb*2),
                                        40: int(grs_na_35_x10fb*4),
                                        60: int(grs_na_35_x10fb*6),
                                        80: int(grs_na_35_x10fb*8),
                                        100: 1123,
                                      }, 
                    'GtoWW35brReco' : { 
                        0: 0,
                        100: 1113, # broad signal. number of signal contamination; len(sig_in_training_nums) == len(signals)
                                    }
                    }


# signals
resonance = 'na'
#signals = ['GtoWW15'+resonance+'Reco', 'GtoWW25'+resonance+'Reco', 'GtoWW35'+resonance+'Reco', 'GtoWW45'+resonance+'Reco']
signals = ['GtoWW35'+resonance+'Reco']
#masses = [1500, 2500, 3500, 4500]
masses = [3500]
# xsecs = [100., 10., 1., 0.]
xsecs = [100, 80, 60, 40, 20, 0]
#xsecs = [0.]
sig_in_training_nums_arr = [[signal_contamin[sig][xsec] for xsec in xsecs] for sig in signals]
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
# quantiles = [0.1, 0.99]
train_split = 0.5 #### **** TODO: update **** ####

# to run
make_qcd_train_test_datasample = True
do_qr = True
train_qr = True
do_bump_hunt = False
model_path_date = '20220303'

Parameters = recordtype('Parameters','run_n_vae, run_n_qr, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, epochs, read_n, qr_model_t')
params = Parameters(run_n_vae=113,
                    run_n_qr=9, #### **** TODO: update **** ####
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrain'+str(int(train_split*100))+'pct', 
                    qcd_test_sample_id='qcdSigAllTest'+str(int((1-train_split)*100))+'pct',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=None,
                    qr_model_t=stco.QR_Model.DENSE) #### **** TODO: update **** ####

print('\n'+'*'*70+'\n'+'\t\t\t TRAINING RUN \n'+str(params)+'\n'+'*'*70)

### paths ###

# data inputs: /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$ 
# data outputs (selections): /eos/user/k/kiwoznia/data/QR_results/events/vae_run_$run_n_vae$/qr_run_$run_n_qr$/sig_GtoWW35naReco/xsec_100/loss_rk5_05
# model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n_vae)})
print('reading samples from {}'.format(paths.base_dir))

if do_qr:
    # if datasets not yet prepared, prepare them, dump and return (same qcd train and testsample for all signals and all xsecs)
    if make_qcd_train_test_datasample:
        qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, train_split=train_split, **cuts.signalregion_cuts)
    # else read from file
    else:
        qcd_train_sample = js.JetSample.from_input_dir(params.qcd_train_sample_id, paths.sample_dir_path(params.qcd_train_sample_id), read_n=params.read_n) 
        qcd_test_sample_ini = js.JetSample.from_input_dir(params.qcd_test_sample_id, paths.sample_dir_path(params.qcd_test_sample_id), read_n=params.read_n)


#****************************************#
#      for each signal: QR & dijet fit
#****************************************#

for sig_sample_id, sig_in_training_nums, mass in zip(signals, sig_in_training_nums_arr, masses):

    params.sig_sample_id = sig_sample_id

    if do_qr:
        sig_sample_ini = js.JetSample.from_input_dir(params.sig_sample_id, paths.sample_dir_path(params.sig_sample_id), **cuts.signalregion_cuts)

    # ************************************************************
    #     for each signal xsec: train and apply QR, do bump hunt 
    # ************************************************************
    for xsec, sig_in_training_num in zip(xsecs, sig_in_training_nums):

        param_dict = {'$run_n_vae$': str(params.run_n_vae), '$run_n_qr$': str(params.run_n_qr), '$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(xsec)), '$loss_strat$': params.strategy_id}
        experiment = ex.Experiment(run_n=params.run_n_vae, param_dict=param_dict).setup(model_dir_qr=True, analysis_dir_qr=True)
        result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path(param_dict) # in selection paths new format with run_x, sig_x, ...

        # ************************************************************
        #                           QR
        # ************************************************************
        if do_qr:

            # create new test samples for new xsec QR (quantile cut results)

            qcd_test_sample = copy.deepcopy(qcd_test_sample_ini)
            sig_sample = copy.deepcopy(sig_sample_ini)
            if train_qr:
                mixed_train_sample, mixed_valid_sample = dapr.inject_signal(qcd_train_sample, sig_sample_ini, sig_in_training_num)
            
            model_paths = []
            
            for quantile in quantiles:

                # using inverted quantile because of dijet fit code
                inv_quant = round((1.-quantile),2)

                # ********************************************
                #               train or load
                # ********************************************

                if train_qr:

                    print('training on {} events, validating on {}'.format(len(mixed_train_sample), len(mixed_valid_sample)))

                    # train and save QR model
                    discriminator = qrwf.train_QR(quantile, mixed_train_sample, mixed_valid_sample, params, qr_model_t=params.qr_model_t)
                    discriminator_path = qrwf.save_QR(discriminator, params, experiment.model_dir_qr, quantile, xsec)
                    model_paths.append(discriminator_path)

                else: # else load discriminators
                    discriminator = qrwf.load_QR(params, experiment, quantile, xsec, model_path_date)
                
                
                # ********************************************
                #               predict
                # ********************************************
                qcd_test_sample = qrwf.predict_QR(discriminator, qcd_test_sample, inv_quant)
                sig_sample = qrwf.predict_QR(discriminator, sig_sample, inv_quant)


            # write results for all quantiles
            print('writing selections to ', result_paths.base_dir)
            qcd_test_sample.dump(result_paths.sample_file_path(params.qcd_test_sample_id, mkdir=True))
            sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id))

            # plot results
            discriminator_list = []
            for q, model_path in zip(quantiles, model_paths):
                discriminator = disc.QRDiscriminator_KerasAPI(quantile=q, loss_strategy=ansc.anomaly_score_strategy_dict[params.strategy_id])
                discriminator.load(model_path)
                discriminator_list.append(discriminator)

            title_suffix = ' trained qcd SR + '+params.sig_sample_id.replace('Reco','')+' at xsec '+str(int(xsec)) + 'fb'
            plot_name = 'multi_discr_cut_'+params.sig_sample_id.replace('Reco','')+'_x'+str(int(xsec))
            andi.analyze_multi_quantile_discriminator_cut(discriminator_list, mixed_valid_sample, title_suffix=title_suffix, plot_name=plot_name, fig_dir=experiment.analysis_dir_qr_cuts)

        # ********************************************
        #               dijet fit
        # ********************************************
        if do_bump_hunt:
            dijet_dir = '/eos/home-k/kiwoznia/dev/vae_dijet_fit/VAEDijetFit'
            runstr = "python run_dijetfit.py --run -n {} -i {} -M {} --sig {}.h5 --sigxsec {} --qcd {}.h5 --res {} --loss {}"
            cmd = runstr.format(params.run_n_vae, result_paths.base_dir, mass, sdfr.path_dict['file_names'][params.sig_sample_id], xsec, sdfr.path_dict['file_names'][params.qcd_test_sample_id], resonance, params.strategy_id)  
            print("running ", cmd)
            subprocess.check_call('pwd && source setupenv.sh && ' + cmd, cwd=dijet_dir, shell=True, executable="/bin/bash")
            print('finished')
