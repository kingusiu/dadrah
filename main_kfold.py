from recordtype import recordtype
import pathlib
import os
import numpy as np

import dadrah.kfold_pipeline.kfold_training as ktrain
import dadrah.kfold_pipeline.kfold_envelope as kenlo
import dadrah.kfold_pipeline.kfold_poly_fitting as kpofi
import dadrah.kfold_pipeline.kfold_prediction as kpred
import dadrah.kfold_pipeline.kfold_string_constants as kstco
import dadrah.kfold_pipeline.kfold_util as kutil
import dadrah.util.logging as log


# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':


    Parameters = recordtype('Parameters','qr_run_n, kfold_n, quantiles, qcd_sample_id, sig_sample_id, sig_xsec, score_strategy_id, read_n, layers_n, nodes_n, epochs, optimizer, batch_sz, lr, env_run_n, binning, poly_run_n, poly_order')
    params = Parameters(qr_run_n=403, 
                        kfold_n=5, 
                        quantiles=[0.9], # [0.3,0.5,0.7,0.9]
                        qcd_sample_id='qcdSigAll', 
                        sig_sample_id='GtoWW35naReco', 
                        sig_xsec=0, 
                        score_strategy_id='rk5_05', 
                        read_n=None, 
                        layers_n=5, 
                        nodes_n=60, 
                        epochs=50, 
                        optimizer='adam', 
                        batch_sz=256, 
                        lr=0.001, 
                        env_run_n=0, 
                        binning='dijet', 
                        poly_run_n=0, 
                        poly_order=11
                        )

    train_models = True
    calc_envelope = False
    fit_polynomials = False
    predict = False


    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t MAIN K-FOLD SCRIPT \n'+str(params)+'\n'+'*'*70)

    ### paths

    # models written to: /eos/home-k/kiwoznia/data/QR_models/vae_run_113/qr_run_$run_n_qr$
    qr_model_dir = kstco.get_qr_model_dir(params)


    if train_models:

        # ****************************************************
        #                   train k models
        # ****************************************************

        tb_base_dir = 'logs/tensorboard/' + str(params.qr_run_n)
        #os.system('rm -rf ' + tb_base_dir)
        
        model_paths = ktrain.train_k_models(params, qr_model_dir, tb_base_dir)

    else:

        model_paths = kutil.get_model_paths(params, qr_model_dir)


    if calc_envelope:

        # ****************************************************
        #                calculate cut envelope
        # ****************************************************


        ### bin edges
        # multiple binning options: dijet, linear, exponential
        bin_edges = kutil.get_dijet_bins()

        envelope_path = kenlo.compute_kfold_envelope(params, model_paths, bin_edges)

    else:

        envelope_path = kstco.get_envelope_dir(params) # load envelope path

    
    if fit_polynomials:

        # ****************************************************
        #                fit polynomials
        # ****************************************************

        polynomial_paths = kpofi.fit_kfold_polynomials(params, envelope_path)

    else:

        polynomial_paths = kstco.get_polynomials_full_file_path(params)



    if predict:

        # ****************************************************
        #                predict background and signal
        # ****************************************************

        selection_path = kpred.predict_with_polynomials(params, polynomial_paths)


# end main    

