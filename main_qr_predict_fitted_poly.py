import numpy as np
from recordtype import recordtype
import os
import json
import pathlib

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.selection.loss_strategy as lost
import dadrah.util.data_processing as dapr
import pofah.phase_space.cut_constants as cuts
import dadrah.util.string_constants as stco
import dadrah.selection.qr_workflow as qrwf




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
    0.99 : np.poly1d([-4.16600751e-18, 4.49261774e-14, -1.60506482e-10, 6.43707025e-07, -5.03695064e-04, 1.87157225e+00]),
    0.9 : np.poly1d([-3.27587188e-18, 3.57458088e-14, -1.17939773e-10, 4.22470378e-07, -2.11205166e-04, 1.65011401e+00]),
    0.1 : np.poly1d([-8.34423542e-19, -2.39966921e-16, 8.92683714e-11, -3.67360323e-07, 9.83976050e-04, 7.67849195e-01]),
    0.7 : np.poly1d([ 4.56721509e-19, -1.29451508e-14,  1.21611625e-10, -2.08025414e-07, 5.29528854e-04,  1.26264581e+00]),
    0.5 : np.poly1d([-7.77474839e-20, -8.90000193e-15,  1.18415519e-10, -2.86784107e-07, 6.98665912e-04,  1.11383712e+00]),
    0.3 : np.poly1d([-1.18038765e-18,  4.05419442e-15,  6.37113446e-11, -2.27612202e-07, 7.11545714e-04,  1.02408896e+00])
}


def inout_paths(params, sig_id, sig_xsec):

    sig_id = sig_id[:-4]

    envelope_json = '/eos/user/k/kiwoznia/data/QR_results/analysis/vae_run_{}/qr_run_{}/sig_{}/xsec_{}/loss_rk5_05/envelope/cut_stats_allQ_{}_xsec_{}.json'.format(
                        params.run_n_vae, params.run_n_qr, sig_id, int(sig_xsec), sig_id, int(sig_xsec))

    result_path = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/QR_models/envelope/fitted_selections/vae_run_{}/qr_run_{}/sig_{}/xsec_{}/loss_rk5_05/order_{}'.format(
                    params.run_n_vae, params.run_n_qr, sig_id, int(sig_xsec), params.poly_order)

    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

    return envelope_json, result_path



#****************************************#
#           run parameters
#****************************************#

# signals
resonance = 'na'
# signals = ['GtoWW15'+resonance+'Reco', 'GtoWW25'+resonance+'Reco', 'GtoWW35'+resonance+'Reco', 'GtoWW45'+resonance+'Reco']
signals = ['GtoWW35'+resonance+'Reco']
#masses = [1500, 2500, 3500, 4500]
masses = [3500]
sig_xsec = 0.
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


Parameters = recordtype('Parameters','run_n_vae, run_n_qr, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, read_n, poly_order')
params = Parameters(run_n_vae=113,
                    run_n_qr=7, 
                    qcd_sample_id='qcdSigReco', 
                    qcd_ext_sample_id='qcdSigExtReco',
                    qcd_train_sample_id='qcdSigAllTrainReco', 
                    qcd_test_sample_id='qcdSigAllTestReco',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    read_n=None,
                    poly_order=5)


sample_paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n_vae)})

#****************************************#
#      for qcd and each signal: 
#   read data, make selection and dump
#****************************************#

for sig_id in signals: # different qr training for each signal injection

    envelope_json, result_path = inout_paths(params, sig_id, sig_xsec)
    polynomials = qrwf.fit_polynomial_from_envelope(envelope_json, quantiles, params.poly_order)

    for sample_id in [params.qcd_test_sample_id, sig_id]: # make prediction for qcd and matching signal sample
        
        sample = js.JetSample.from_input_dir(sample_id, sample_paths.sample_dir_path(sample_id), **cuts.signalregion_cuts)

        for quantile in quantiles:

            # using inverted quantile because of dijet fit code
            inv_quant = round((1.-quantile),2)

            #print('predicting {}'.format(sample.name))
            selection = fitted_selection(sample, params.strategy_id, quantile, polynomials)
            sample.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)

        # write results for all quantiles
        file_path = os.path.join(result_path, sdfr.path_dict['file_names'][sample_id]+'.h5')
        sample.dump(file_path)


    