import os
from recordtype import recordtype
import pathlib
import copy

import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import dadrah.util.data_processing as dapr
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


