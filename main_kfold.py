from recordtype import recordtype
import pathlib
import os
import numpy as np
import argparse

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

    # command line
    parser = argparse.ArgumentParser(description='read arguments for k-fold QR training')
    parser.add_argument('-r', dest='qr_run_n', type=int, help='experiment run number')
    # qr hyperparams
    parser.add_argument('-ln', dest='layers_n', type=int, help='number of layers')
    parser.add_argument('-nn', dest='nodes_n', type=int, help='number of nodes')
    parser.add_argument('-bn', dest='batch_sz', type=int, help='batch size')
    parser.add_argument('-lr', dest='lr', type=float, help='learning rate')
    parser.add_argument('-ac', dest='acti', type=str, help='activation function')
    parser.add_argument('-in', dest='initial', choices=['he','glorot'], help='weight initializer')
    # samples
    parser.add_argument('-read', dest='read_n', type=int, help='number of samples to read')
    # envelope and polynomials run n
    parser.add_argument('-en', dest='env_run_n', type=int, help='envelope number (f of bins)', default=0)
    parser.add_argument('-pn', dest='poly_run_n', type=int, help='polyfit number (f of order)', default=0)
    # loading options
    parser.add_argument('--loadqr', dest='train_models', action="store_false", help='load previously trained qr models')
    parser.add_argument('--loadenv', dest='calc_envelope', action="store_false", help='load previously calulated envelope')
    parser.add_argument('--loadpoly', dest='fit_polynomials', action="store_false", help='load previously fitted polynomials')
    parser.add_argument('--siginject', dest='sig_inject', action='store_true', help='inject signal into qr training')
    # binning options
    parser.add_argument('-bi', dest='binning', choices=['linear', 'expo', 'dijet'], help='binning basis for envelope', default='dijet')
    parser.add_argument('-bis', dest='bin_start', type=int, help='index of first bin')
    parser.add_argument('-bin', dest='n_bins', type=int, help='total number of bins')
    parser.add_argument('-bimi', dest='min_mjj', type=float, help='maximal mjj')
    parser.add_argument('-bima', dest='max_mjj', type=float, help='minimal mjj')
    parser.add_argument('--bie', dest='bin_centers', action='store_false')
    # signal injection
    parser.add_argument('--siginj', dest='sig_injected', help='inject signal (at 100fb)', action='store_true')
    parser.add_argument('-siid', dest='sig_sample_id', choices=['GtoWW15na', 'GtoWW25na', 'GtoWW35na', 'GtoWW45na', 'GtoWW15br', 'GtoWW25br'], default='GtoWW35na')

    args = parser.parse_args()
    # optional binning kwargs
    kwargs_bins = {k:v for k,v in vars(args).items() if k in ['bin_start','bin_centers','n_bins','min_mjj','max_mjj'] and v is not None} 

    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*60+'\n'+'\t\t\t PREDICTION RUN \n'+str(args)+'\n'+'*'*60)


    # fixed 
    Parameters = recordtype('Parameters','qr_run_n, kfold_n, quantiles, qcd_sample_id, sig_sample_id, sig_xsec, score_strategy_id, read_n, layers_n, nodes_n, batch_sz, acti, initial, lr, epochs, optimizer, reg_coeff, env_run_n, binning, poly_run_n, poly_order')
    params = Parameters(qr_run_n=args.qr_run_n,
                        kfold_n=5, 
                        quantiles=[0.3,0.5,0.7,0.9],
                        qcd_sample_id='qcdSigAll', 
                        sig_sample_id=args.sig_sample_id+'Reco', 
                        sig_xsec=(100 if args.sig_injected else 0), 
                        score_strategy_id='rk5_05', 
                        read_n=args.read_n,
                        layers_n=args.layers_n,
                        nodes_n=args.nodes_n,
                        batch_sz=args.batch_sz,
                        acti=args.acti,
                        initial=(args.initial+'_uniform' if args.initial is not None else None),
                        lr=args.lr, 
                        epochs=50, 
                        optimizer='adam',
                        reg_coeff=0., 
                        env_run_n=args.env_run_n, 
                        binning=args.binning, 
                        poly_run_n=args.poly_run_n, 
                        poly_order=11,
                        )

    predict = True


    logger.info('\n'+'*'*70+'\n'+'\t\t\t MAIN K-FOLD SCRIPT \n'+str(params)+'\n'+'*'*70)


    if args.train_models:

        # ****************************************************
        #                   train k models
        # ****************************************************

        logger.info('training QR model ' + str(args.qr_run_n))

        tb_base_dir = 'logs/tensorboard/' + str(args.qr_run_n)
        
        # models written to: /eos/home-k/kiwoznia/data/QR_models/vae_run_113/qr_run_$run_n_qr$
        # import ipdb; ipdb.set_trace()
        model_paths = ktrain.train_k_models(params, tb_base_dir)

    else:

        logger.info('loading QR model ' + str(args.qr_run_n))

        model_paths = kutil.get_model_paths(params)


    if args.calc_envelope:

        # ****************************************************
        #                calculate cut envelope
        # ****************************************************


        ### bin edges
        # multiple binning options: dijet, linear, exponential
        bin_edges = kutil.get_bins(bin_type=params.binning, **kwargs_bins)
        logger.info('calculating envelope ' +str(params.env_run_n)+ ' with bins ' + ','.join(['{:.2f}'.format(b) for b in bin_edges]))

        envelope_path = kenlo.compute_kfold_envelope(params, model_paths, bin_edges)

    else:

        logger.info('loading envelope nr ' +str(params.env_run_n))
        envelope_path = kstco.get_envelope_dir(params) # load envelope path

    
    if args.fit_polynomials:

        # ****************************************************
        #                fit polynomials
        # ****************************************************

        logger.info('fitting polynomials nr '+str(params.poly_run_n)+' of order '+str(params.poly_order))
        polynomial_paths = kpofi.fit_kfold_polynomials(params, envelope_path)

    else:
        logger.info('loading polynomials nr '+str(params.poly_run_n))
        polynomial_paths = kstco.get_polynomials_full_file_path(params)



    if predict:

        # ****************************************************
        #                predict background and signal
        # ****************************************************

        selection_path = kpred.predict_with_polynomials(params, polynomial_paths)


# end main    

