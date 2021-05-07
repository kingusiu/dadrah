import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf
from recordtype import recordtype

import dadrah.util.data_processing as dapr


# setup runtime params and csv file
quantiles = [0.1, 0.9, 0.99]

Parameters = recordtype('Parameters','run_n, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, strategy_id, epochs, read_n')
params = Parameters(run_n=113, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrainReco', 
                    qcd_test_sample_id='qcdSigAllTestReco',
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=None)

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})



# split qcd data

# for each quantile

    # for each qcd data part

        # train qr

        # save qr

        # predict cut values per bin

        # store cut values to csv file

# plot quantile cut bands