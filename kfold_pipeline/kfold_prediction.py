import os
import pathlib

import pofah.jet_sample as jesa
import pofah.phase_space.cut_constants as cuco
import dadrah.util.logging as log
import dadrah.util.data_processing as dapr
import dadrah.selection.anomaly_score_strategy as ansc 
import dadrah.kfold_pipeline.kfold_util as kutil
import dadrah.kfold_pipeline.kfold_string_constants as kstco



logger = log.get_logger(__name__)



def predict(sample, strategy_id, quantile, polynomials, x_shift=0.):

    loss_strategy = ansc.an_score_strategy_dict[strategy_id]
    loss = loss_strategy(sample)
    loss_cut = polynomials[quantile]
    
    return loss > loss_cut(sample['mJJ']-x_shift) # shift x by min mjj for bias fixed lmfits


def predict_with_polynomials(params, polys_path):
    
    #****************************************#
    #           read in polynomials
    #****************************************#

    logger.info('reading polynomials from ' + polys_path)
    polynomials_folds, x_shift = dapr.read_polynomials_from_json(polys_path, params.quantiles, params.kfold_n)

    #****************************************#
    #           read in all qcd data
    #****************************************#

    qcd_sample_parts = kutil.read_kfold_datasets(kstco.vae_out_dir_kfold_qcd, params.kfold_n, read_n=params.read_n)

    #*********************************************************************#
    #      predict & write: make selections from fitted polynomials
    #*********************************************************************#

    output_path = kstco.get_polynomials_out_data_dir(params)

    ### predict and write qcd

    for k, sample in zip(range(1, params.kfold_n+1), qcd_sample_parts):
        for quantile in params.quantiles:
            selection = predict(sample, params.score_strategy_id, quantile, polynomials_folds['fold_{}'.format(k)], x_shift)
            sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)

    qcd_sample_results = qcd_sample_parts[0]
    for k in range(1, params.kfold_n):
        qcd_sample_results = qcd_sample_results.merge(qcd_sample_parts[k])

    qcd_sample_results.dump(os.path.join(output_path,params.qcd_sample_id+'.h5'))


    #### predict and write signals

    sig_kfold_n = params.kfold_n+1
    sig_sample_ids = [params.sig_sample_id] # todo: change for multisignal prediction when sig_xsec=0

    for sig_sample_id in sig_sample_ids:

        sample_path_sig = os.path.join(kstco.vae_out_dir, kstco.vae_out_sample_dir_dict[sig_sample_id])
        logger.info('applying polynomial cuts for selection of ' + sample_path_sig)

        sig_sample = jesa.JetSample.from_input_dir(sig_sample_id, sample_path_sig, read_n=params.read_n, **cuco.signalregion_cuts)

        for quantile in params.quantiles:
            selection = predict(sig_sample, params.score_strategy_id, quantile, polynomials_folds['fold_{}'.format(sig_kfold_n)], x_shift)
            sig_sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)

        sig_sample.dump(os.path.join(output_path,params.sig_sample_id+'.h5'))

    return output_path


def predict_with_envelope():
    pass