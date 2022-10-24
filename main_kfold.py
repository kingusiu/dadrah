from recordtype import recordtype

import dadrah.kfold_pipeline.kfold_training as ktrain



# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':


    Parameters = recordtype('Parameters','qr_run_n, kfold_n, quantiles')
    params = Parameters(qr_run_n=400,
                        kfold_n=5,
                        quantiles=[0.3, 0.5, 0.7, 0.9]
                        )

    # ****************************************************
    #                   train k models

    # models written to: /eos/home-k/kiwoznia/data/QR_models/vae_run_113/qr_run_$run_n_qr$
    
    ktrain.train_k_models(params) # return model paths?

    