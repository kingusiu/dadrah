from recordtype import recordtype
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import tensorflow as tf
import mplhep as hep
import matplotlib.cm as cm

import dadrah.kfold_pipeline.kfold_poly_fitting as kpofi
import dadrah.kfold_pipeline.kfold_string_constants as kstco
import dadrah.kfold_pipeline.kfold_util as kutil
import dadrah.util.logging as log
import dadrah.selection.anomaly_score_strategy as ansc 
import pofah.jet_sample as jesa
import dadrah.selection.quantile_regression as qure
import vande.vae.layers as layers



def plot_model_cuts(params, models_fold, sample, score_strategy, k, fig_dir, xlim=False):

    # Load CMS style sheet
    #plt.style.use(hep.style.CMS)

    # setup colormap for 2dhist
    cmap = cm.get_cmap('Blues')
    xc = np.linspace(0.0, 1.0, 100)
    color_list = cmap(xc)
    color_list = np.vstack((color_list[0], color_list[40:])) # keep white, drop light colors 
    my_cm = colors.ListedColormap(color_list)

    # setup colors for cut lines
    palette = ['#3E96A1', '#FF9505', '#EC4E20', '#9E0059', '#713E5A' ][1:] # skip first because 2d background hist = blue
    palette.reverse()
    feature_key = 'mJJ'
    
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])
    x_max = np.max(sample[feature_key])
    an_score = score_strategy(sample)
    x_top = np.percentile(sample[feature_key], 99.999) if xlim else x_max
    x_range = ((x_min * 0.9,x_top), (np.min(an_score), np.percentile(an_score, 99.99)))
    plt.hist2d((sample[feature_key]), an_score, range=x_range, norm=(LogNorm()), bins=200, cmap=my_cm, cmin=0.001)
    xs = np.arange(x_min, x_max, 0.001 * (x_max - x_min))
    for q, c in zip(params.quantiles, palette):
        model = models_fold[q]
        cuts = np.squeeze(model.predict(xs)) #np.squeeze(model.predict([xs,xs]))
        plt.plot(xs, cuts, '-', lw=2.5, label='Q '+str(int(q*100))+'%', color=c)
    plt.ylabel('min(L1,L2)')
    plt.xlabel('$M_{jj}$ [GeV]')
    #plt.title('quantile cuts' + title_suffix)
    plt.colorbar()
    plt.legend(loc='best', title='quantile cuts')
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, 'qr_cuts_fold'+ str(k) +'_allQ.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)



# ******************************************** #
#                    main                      #
# ******************************************** #


if __name__ == '__main__':


    Parameters = recordtype('Parameters','qr_run_n, kfold_n, quantiles, qcd_sample_id, sig_sample_id, sig_xsec, score_strategy_id, read_n, layers_n, nodes_n, epochs, optimizer, batch_sz, lr, env_run_n, binning, poly_run_n, poly_order')
    params = Parameters(qr_run_n=32, 
                        kfold_n=5, 
                        quantiles=[0.3,0.5,0.7,0.9], # [0.9]
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

    plot_models = True
    plot_envelope = False
    plot_polynomials = False
    plot_predict = False


    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t MAIN K-FOLD PLOTTING SCRIPT \n'+str(params)+'\n'+'*'*70)

    ### paths

    # models written to: /eos/home-k/kiwoznia/data/QR_models/vae_run_113/qr_run_$run_n_qr$
    qr_model_dir = kstco.get_qr_model_dir(params)

    #****************************************#
    #           read in all qcd data
    #****************************************#

    qcd_sample_parts = kutil.read_kfold_datasets(kstco.vae_out_dir_kfold_qcd, params.kfold_n, read_n=params.read_n)
    score_strategy = ansc.an_score_strategy_dict[params.score_strategy_id]


    if plot_models:

        # ****************************************************
        #          plot multi-quantile cuts per fold
        # ****************************************************

        model_paths = kutil.get_model_paths(params, qr_model_dir)
        fig_dir = kstco.get_qr_model_fig_dir(params)
        logger.info('plotting QR model cuts per fold to ' + fig_dir)

        for k, qcd_sample_part in zip(range(1,params.kfold_n+1), qcd_sample_parts):

            qcd_train, qcd_valid = jesa.split_jet_sample_train_test(qcd_sample_part, frac=0.7)

            models_fold = {}
            for q in params.quantiles:
                q_str = 'q'+str(int(q*100))
                model_path = model_paths[q_str]['fold' + str(k)]
                model = tf.keras.models.load_model(model_path, custom_objects={'QrModel': qure.QrModel, 'StdNormalization': layers.StdNormalization}, compile=False)
                models_fold[q] = model
        
            plot_model_cuts(params, models_fold, qcd_sample_part, score_strategy, k, fig_dir, xlim=True)


        # end for each fold of k

    if plot_envelope:

        # ****************************************************
        #                calculate cut envelope
        # ****************************************************


        ### bin edges
        # multiple binning options: dijet, linear, exponential
        bin_edges = kutil.get_dijet_bins()

        envelope_path = kstco.get_envelope_dir(params) # load envelope path

    
    if plot_polynomials:

        # ****************************************************
        #                fit polynomials
        # ****************************************************

        polynomial_paths = kpofi.fit_kfold_polynomials(params, envelope_path)

    else:

        polynomial_paths = kstco.get_polynomials_full_file_path(params)



    if plot_predict:

        # ****************************************************
        #                predict background and signal
        # ****************************************************

        selection_path = kpred.predict_with_polynomials(params, polynomial_paths)


# end main    

