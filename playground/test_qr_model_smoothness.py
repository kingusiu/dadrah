from recordtype import recordtype

import dadrah.util.run_paths as runpa
import dadrah.selection.loss_strategy as lost
import dadrah.util.string_constants as stco
import dadrah.util.logging as log


if __name__ == '__main__':

    train_split = 0.3

    Parameters = recordtype('Parameters','vae_run_n, qr_run_n, qcd_sample_id, qcd_ext_sample_id, qcd_train_sample_id, qcd_test_sample_id, sig_sample_id, strategy_id, epochs, read_n, qr_model_t')
    params = Parameters(
                    vae_run_n=113,
                    qr_run_n=111,
                    qcd_train_sample_id='qcdSigAllTrain'+str(int(train_split*100))+'pct', 
                    qcd_test_sample_id='qcdSigAllTest'+str(int((1-train_split)*100))+'pct',
                    sig_sample_id='GtoWW35naReco',
                    strategy_id='rk5_05',
                    epochs=100,
                    read_n=None,
                    qr_model_t=stco.QR_Model.DENSE,
                    )

    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t train QR \n'+str(params)+'\n'+'*'*70)

    quantiles = [0.3, 0.5, 0.7, 0.9]
    # quantiles = [0.9]
    sig_xsec = 0.
    loss_strategy = lost.loss_strategy_dict[params.strategy_id]
    in_path_ext_dict = {'vae_run': str(vae_run_n), 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Train_signalregion_parts': None}

    ### paths ###

    # data inputs (mjj & vae-scores): /eos/user/k/kiwoznia/data/VAE_results/events/run_$vae_run_n$
    # model outputs: /eos/home-k/kiwoznia/data/QR_models/vae_run_$run_n_vae$/qr_run_$run_n_qr$
    # data outputs (selections [0/1] per quantile): /eos/user/k/kiwoznia/data/QR_results/events/

    paths = runpa.RunPaths(in_data_dir=strc.dir_path_dict['base_dir_vae_results'], in_data_names=strc.file_name_path_dict, out_data_dir=strc.dir_path_dict['base_dir_qr_selections'])
    paths.extend_in_path_data(in_path_ext_dict)

    #****************************************#
    #           read in qcd data
    #****************************************#

    logger.info('reading samples from {}'.format(paths.base_dir))
