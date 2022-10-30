
import dadrah.util.logging as log
import dadrah.util.data_processing as dapr


logger = log.get_logger(__name__)



def predict_with_polynomials(params, polys_path):
    
    #****************************************#
    #           read in polynomials
    #****************************************#

    logger.info('reading polynomials from ' + polys_path)
    polynomials_folds = dapr.read_polynomials_from_json(polys_path, params.quantiles, params.kfold_n)

    #****************************************#
    #           read in all qcd data
    #****************************************#

    qcd_sample_parts = kutil.read_kfold_datasets(kstco.input_dir_kfold, params.kfold_n, read_n=params.read_n)

    #************************************************************#
    #      predict & write: make selections from fitted polynomials
    #************************************************************#


    ### predict and write qcd

    for k, sample in zip(range(1, params.kfold_n+1), qcd_sample_parts):
        for quantile in params.quantiles:
            selection = qrwf.fitted_selection(sample, params.strategy_id, quantile, polynomials_folds['fold_{}'.format(k)], params.xshift)
            sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)

    qcd_sample_results = qcd_sample_parts[0]
    for k in range(1, params.kfold_n):
        qcd_sample_results = qcd_sample_results.merge(qcd_sample_parts[k])

    qcd_sample_results.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True))


    #### predict and write signals

    sig_kfold_n = params.kfold_n+1

    for sig_sample_id in sig_sample_ids:

        param_dict = {'$vae_run_n$': str(params.vae_run_n), '$qr_run_n$': str(params.qr_run_n), '$sig_name$': sig_sample_id, '$sig_xsec$': str(int(sig_xsec)), '$loss_strat$': params.strategy_id}
        result_paths = safa.SamplePathDirFactory(sdfs.path_dict).update_base_path(param_dict) # in selection paths new format with run_x, sig_x, ...

        logger.info('applying discriminator cuts for selection')

        sig_sample = jesa.JetSample.from_input_dir(sig_sample_id, input_paths_sig.sample_dir_path(sig_sample_id), read_n=params.read_n, **cuco.signalregion_cuts)

        for quantile in quantiles:
            selection = qrwf.fitted_selection(sig_sample, params.strategy_id, quantile, polynomials_folds['fold_{}'.format(sig_kfold_n)], params.xshift)
            sig_sample.add_feature('sel_q{:02}'.format(int(quantile*100)), selection)

        sig_sample.dump(result_paths.sample_file_path(sig_sample_id, mkdir=True))



def predict_with_envelope():
    pass