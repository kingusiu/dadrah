import os
from recordtype import recordtype
import numpy as np
import pathlib

import pofah.jet_sample as jesa
import pofah.util.sample_factory as safa
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import pofah.phase_space.cut_constants as cuco
import dadrah.util.string_constants as stco
import dadrah.util.logging as log
import dadrah.util.data_processing as dapr
import dadrah.selection.loss_strategy as lost



def slice_datasample_n_parts(data, parts_n, shuffle=False):
    if shuffle:
        data = data.shuffle()
    cuts = np.linspace(0, len(data), num=parts_n+1, endpoint=True, dtype=int)
    return [data.filter(slice(start,stop)) for (start, stop) in zip(cuts[:-1], cuts[1:])]


#****************************************#
#           params
#****************************************#

# data inputs: /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$ 
# data outputs (5 fold subsets): /eos/user/k/kiwoznia/data/VAE_results/events/run_$run_n_vae$/5folds 

Parameters = recordtype('Parameters','vae_run_n, qcd_sample_id, qcd_ext_sample_id, sig_sample_id, strategy_id, read_n, kfold_n, shuffle')
params = Parameters(vae_run_n=113,
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    read_n=None,
                    kfold_n=5,
                    shuffle=True
                    ) 

loss_function = lost.loss_strategy_dict[params.strategy_id]


# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t making k-fold datasets \n'+str(params)+'\n'+'*'*70)

#****************************************#
#           read in all qcd data
#****************************************#
input_paths = safa.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.vae_run_n)})
output_dir = '/eos/user/k/kiwoznia/data/VAE_results/events/run_113/qcd_sqrtshatTeV_13TeV_PU40_NEW_'+str(params.kfold_n)+'fold_signalregion_parts'
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


qcd_sample_all = dapr.merge_qcd_base_and_ext_datasets(params, input_paths, **cuco.signalregion_cuts)
logger.info('qcd all: min mjj = {}, max mjj = {}'.format(np.min(qcd_sample_all['mJJ']), np.max(qcd_sample_all['mJJ'])))
logger.info('qcd all: min loss = {}, max loss = {}'.format(np.min(loss_function(qcd_sample_all)), np.max(loss_function(qcd_sample_all))))
# split qcd data
qcd_sample_parts = slice_datasample_n_parts(qcd_sample_all, params.kfold_n, shuffle=params.shuffle)

for k, qcd_sample_part in enumerate(qcd_sample_parts):
    qcd_sample_part.dump(os.path.join(output_dir,'qcd_sqrtshatTeV_13TeV_PU40_NEW_fold'+str(k+1)+'.h5'))
