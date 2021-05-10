import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype
import numpy as np

import dadrah.util.data_processing as dapr
import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_reco as sdfr


def slice_datasample_n_parts(data, parts_n):
    cuts = np.linspace(0, len(data), num=parts_n+1, endpoint=True, dtype=int)
    return [data.cut(slice(start,stop)) for (start, stop) in zip(cuts[:-1], cuts[1:])]


# setup runtime params and csv file
quantiles = [0.1, 0.9, 0.99]
parts_n = 5

Parameters = recordtype('Parameters','run_n, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, strategy_id, epochs, read_n')
params = Parameters(run_n=113, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrainReco', 
                    qcd_test_sample_id='qcdSigAllTestReco',
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=int(1e5))

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})
data_qcd_all = dapr.merge_qcd_base_and_ext_datasets(params, paths)
# split qcd data
data_qcd_parts = slice_datasample_n_parts(data_qcd_all, parts_n)

# for each quantile
for quantile in quantiles:

    # for each qcd data part
    for dat_train, dat_valid in zip(data_qcd_parts, data_qcd_parts[1:] + [data_qcd_parts[0]])
        # train qr
        print('training on {} events, validating on {}'.format(len(dat_train), len(dat_valid)))
        
        # save qr

        # predict cut values per bin

        # store cut values to csv file

# plot quantile cut bands