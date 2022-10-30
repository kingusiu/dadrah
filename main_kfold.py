from recordtype import recordtype
import pathlib
import os
import numpy as np

import dadrah.kfold_pipeline.kfold_training as ktrain
import dadrah.kfold_pipeline.kfold_envelope as kenlo
import dadrah.kfold_pipeline.kfold_poly_fitting as kpofi
import dadrah.kfold_pipeline.kfold_string_constants as kstco
import dadrah.kfold_pipeline.kfold_util as kutil


# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':


    Parameters = recordtype('Parameters','qr_run_n, kfold_n, quantiles, sig_sample_id, sig_xsec, score_strategy_id, read_n, layers_n, nodes_n, epochs, batch_sz, lr, env_run_n, poly_run_n, poly_order')
    params = Parameters(qr_run_n=401,
                        kfold_n=5,
                        quantiles=[0.9], # 0.3, 0.5, 0.7, 
                        sig_sample_id='GtoWW35naReco',
                        sig_xsec=0,
                        score_strategy_id='rk5_05',
                        read_n=None,
                        layers_n=4,
                        nodes_n=5,
                        epochs=50,
                        batch_sz=32,
                        lr=1e-3,
                        env_run_n=4020,
                        poly_run_n=40200,
                        poly_order=11
                        )

    train_models = True
    calc_envelope = False
    fit_polynomials = False
    predict = False


    ### paths

    # models written to: /eos/home-k/kiwoznia/data/QR_models/vae_run_113/qr_run_$run_n_qr$
    qr_model_dir = '/eos/home-k/kiwoznia/data/QR_models/vae_run_' + str(kstco.vae_run_n) + '/qr_run_' + str(params.qr_run_n)
    pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)


    if train_models:

        # ****************************************************
        #                   train k models
        # ****************************************************

        fig_dir = 'fig/qr_run_' + str(params.qr_run_n)
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
        tb_base_dir = 'logs/tensorboard/' + str(params.qr_run_n)
        os.system('rm -rf ' + tb_base_dir)
        
        model_paths = ktrain.train_k_models(params, qr_model_dir, fig_dir, tb_base_dir)

    else:

        model_paths = kutil.get_model_paths(params, qr_model_dir)


    if calc_envelope:

        # ****************************************************
        #                calculate cut envelope
        # ****************************************************


        ### bin edges
        # multiple binning options: dijet, linear, exponential
        bin_edges = np.array([1200, 1255, 1320, 1387, 1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206, 
                            2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928, 
                            4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951, 6173, 6402, 6638, 6882]).astype('float')

        envelope_path = kenlo.compute_kfold_envelope(params, model_paths, bin_edges)

    else:

        envelope_path = kutil.get_envelope_path(params) # load envelope path

    
    if fit_polynomials:

        # ****************************************************
        #                fit polynomials
        # ****************************************************

        polynomial_paths = kpofi.fit_kfold_polynomials(params, envelope_path)

    else:

        polynomial_paths = kutil.get_polys_json_path(params)



    if predict:

        # ****************************************************
        #                predict background and signal
        # ****************************************************

        pass

    else:

        pass        

