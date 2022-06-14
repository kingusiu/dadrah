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
import dadrah.selection.loss_strategy as lost
import dadrah.selection.qr_workflow as qrwf
import analysis.analysis_discriminator as andi
import dadrah.util.data_processing as dapr
import dadrah.util.string_constants as stco
import pofah.phase_space.cut_constants as cuts


#****************************************#
#           set runtime params
#****************************************#


train_split = 0.3 #### **** TODO: update **** ####

Parameters = recordtype('Parameters','run_n_vae, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, read_n')
params = Parameters(run_n_vae=113,
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrain'+str(int(train_split*100))+'pct', 
                    qcd_test_sample_id='qcdSigAllTest'+str(int((1-train_split)*100))+'pct',
                    read_n=None,
                    )

print('\n'+'*'*70+'\n'+'\t\t\t making qcd train test datasets \n'+str(params)+'\n'+'*'*70)

### paths ###

# data inputs: /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$ 
# data outputs (split test & train dataset): /eos/user/k/kiwoznia/data/VAE_results/events/run_113/qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Train_signalregion_parts & /eos/user/k/kiwoznia/data/VAE_results/events/run_113/qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_signalregion_parts

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n_vae)})
print('reading samples from {}'.format(paths.base_dir))

qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, train_split=train_split, **cuts.signalregion_cuts)


