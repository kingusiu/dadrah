import numpy as np
from recordtype import recordtype

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.selection.loss_strategy as lost
import dadrah.util.data_processing as dapr
import pofah.phase_space.cut_constants as cuts



#****************************************#
#       fitted polynomial selections
#****************************************#

# 3-param fit

cut_polys_par3 = {
    
    0.99 : np.poly1d([-7.03032980e-11, 1.06542864e-06, -2.17139299e-03, 3.31447885e+00]),
    0.9 : np.poly1d([-4.60568395e-11, 7.63471757e-07, -1.55537204e-03, 2.81116723e+00]),
    0.1 : np.poly1d([-1.57031061e-11, 3.01628110e-07, -5.03564141e-04, 1.83966064e+00])
}

# 5-param-fit

cut_polys_par5 = {
    
    0.99 : np.poly1d([6.39477740e-18, -1.32096520e-13, 9.60377076e-10, -2.68629790e-06, 4.11271170e-03, -5.10765776e-01]),
    0.9 : np.poly1d([4.53587326e-18, -9.43590874e-14, 6.95373175e-10, -1.95383392e-06, 3.02479614e-03, 8.16766148e-03]),
    0.1 : np.poly1d([1.43916815e-18, -3.34572370e-14, 2.74613208e-10, -8.59156457e-07, 1.60246515e-03, 4.73345295e-01])
}

def fitted_selection(sample, strategy_id, quantile, params_n=5):
    loss_strategy = lost.loss_strategy_dict[strategy_id]
    loss = loss_strategy(sample)
    loss_cut = cut_polys_par5[quantile] if params_n == 5 else cut_polys_par3[quantile]
    return loss > loss_cut(sample['mJJ'])





#****************************************#
#           run parameters
#****************************************#

# signals
resonance = 'na'
signals = ['GtoWW15'+resonance+'Reco', 'GtoWW25'+resonance+'Reco', 'GtoWW35'+resonance+'Reco', 'GtoWW45'+resonance+'Reco']
masses = [1500, 2500, 3500, 4500]
xsec = 0.
quantiles = [0.1, 0.9, 0.99]



Parameters = recordtype('Parameters','run_n, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, read_n')
params = Parameters(run_n=113, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrainReco', 
                    qcd_test_sample_id='qcdSigAllTestReco',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    read_n=None)


paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})

#****************************************#
#      for qcd and each signal: 
#   read data, make selection and dump
#****************************************#

for sample_id in [params.qcd_test_sample_id] + signals:

    sample = js.JetSample.from_input_dir(sample_id, paths.sample_dir_path(sample_id), **cuts.signalregion_cuts)

    param_dict = {'$sig_name$': sample_id, '$sig_xsec$': str(int(xsec)), '$loss_strat$': params.strategy_id}
    experiment = ex.Experiment(run_n=params.run_n, param_dict=param_dict).setup(model_dir_qr=True, analysis_dir_qr=True)
    result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(params.run_n), **param_dict}) # in selection paths new format with run_x, sig_x, ...
    result_paths = result_paths.extend_base_path('fitted_cut', 'param5')

    for quantile in quantiles:

        # using inverted quantile because of dijet fit code
        inv_quant = round((1.-quantile),2)

        #print('predicting {}'.format(sample.name))
        selection = fitted_selection(sample, params.strategy_id, quantile)
        sample.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)

        # write results for all quantiles
        print('writing selections to ', result_paths.base_dir)
        qcd_test_sample.dump(result_paths.sample_file_path(params.qcd_test_sample_id, mkdir=True))
        sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id))

    