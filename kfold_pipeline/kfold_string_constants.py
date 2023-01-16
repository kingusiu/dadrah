import pathlib
import os
import glob


# categories
# qr - in - data
# qr - out - models (quantile regressions)
# envelope - in - models (quantile regressions)
# envelope - out - models (envelope)
# polys - in - models (envelope)
# polys - out - models (polynomials)
# polys - in - data
# polys - out - data



# fixed vae
vae_run_n = 113

# *********************************************************** #
#                           IN/OUT DATA                       #
# *********************************************************** #


# qr - in - data
# polys - in - data

vae_out_dir = '/eos/user/k/kiwoznia/data/VAE_results/events/run_'+str(vae_run_n)
vae_out_dir_kfold_qcd = vae_out_dir+'/qcd_sqrtshatTeV_13TeV_PU40_NEW_5fold_signalregion_parts'

vae_out_sample_dir_dict = {
    'GtoWW35naReco' : 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_NEW_parts'
} 

# polys - out - data (polynomials cut outputs with 'sel' columns per quantile)
# => directory structure: qr_run_x/env_run_Y/poly_run_Z
polys_out_data_base_dir = '/eos/user/k/kiwoznia/data/QR_results/events'

def get_polynomials_out_data_dir(params):
    data_dir = os.path.join(polys_out_data_base_dir,'qr_run_'+str(int(params.qr_run_n)),'env_run_'+str(int(params.env_run_n)),'poly_run_'+str(int(params.poly_run_n)))
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


# *********************************************************** #
#                         IN/OUT MODELS                       #
# *********************************************************** #


# QR model -> envelope ( f(binning) ) -> polynomials ( f(order) )
# => directory structure: qr_run_x/env_run_Y/poly_run_Z


# qr k models trained base dir
qr_out_model_dir = '/eos/user/k/kiwoznia/data/QR_results/models'


# qr - out - models (quantile regressions)
# envelope - in - models (quantile regressions)
# e.g.: /eos/user/k/kiwoznia/data/QR_results/models/qr_run_401/QRmodel_run_401_q90_GtoWW35naReco_sigx0_fold4.h5
def get_qr_model_dir(params):
    qr_model_dir = os.path.join(qr_out_model_dir,'qr_run_'+str(int(params.qr_run_n)))
    pathlib.Path(qr_model_dir).mkdir(parents=True, exist_ok=True)
    return qr_model_dir

def get_qr_model_file_name(params,q,k):
    return 'QRmodel_run_'+str(int(params.qr_run_n))+'_q'+str(int(q*100))+'_'+params.sig_sample_id+'_sigx'+str(int(params.sig_xsec))+'_fold'+str(int(k))+'.h5'


# envelope dir, based on qr model, function of binning
# e.g.: /eos/user/k/kiwoznia/data/QR_results/models/qr_run_401/env_run_0/envelope_run_0_allQ_GtoWW35naReco_xsec_0_dijetBin_fold1.json
def get_envelope_dir(params):
    env_out_dir = os.path.join(get_qr_model_dir(params),'env_run_'+str(int(params.env_run_n)))
    pathlib.Path(env_out_dir).mkdir(parents=True, exist_ok=True)
    return env_out_dir


def get_envelope_file_name(params,k):
    return 'envelope_run_'+str(int(params.env_run_n))+'_qr_run_'+str(int(params.qr_run_n))+'_allQ_'+params.sig_sample_id+'_sigx'+str(int(params.sig_xsec))+'_'+params.binning+'Bin'+'_fold'+str(int(k))+'.json'


def get_envelope_file(params,k):
    return glob.glob(get_envelope_dir(params)+'/*'+params.sig_sample_id+'*fold'+str(k)+'.json')[0]


# polynomials dir, based on envelope, function of order
# e.g. /eos/user/k/kiwoznia/data/QR_results/models//qr_run_402/env_run_0/poly_run_0/polynomials_allQ_allFolds_GtoWW35naReco_xsec_0.json
def get_polynomials_dir(params):
    polys_out_dir = os.path.join(get_envelope_dir(params),'poly_run_'+str(int(params.poly_run_n)))
    pathlib.Path(polys_out_dir).mkdir(parents=True, exist_ok=True)
    return polys_out_dir

def get_polynomials_file_name(params):
    return 'polynomials_run_'+str(int(params.poly_run_n))+'_envelope_run_'+str(int(params.env_run_n))+'_qr_run_'+str(int(params.qr_run_n))+'_allQ_'+params.sig_sample_id+'_sigx'+str(int(params.sig_xsec))+'_ord'+str(int(params.poly_order))+'_allFolds.json'

def get_polynomials_full_file_path(params):
    return os.path.join(get_polynomials_dir(params), get_polynomials_file_name(params))


# *********************************************************** #
#                           OUT FIGURES                       #
# *********************************************************** #

# qr models (quantile cuts)
def get_qr_model_fig_dir(params):
    fig_dir = 'fig/qr_run_' + str(params.qr_run_n)+'/qr'
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
    return fig_dir

# envelope (mean cuts with stderr bands, uncertainties over kfolds)
def get_envelope_fig_dir(params):
    fig_dir = 'fig/qr_run_' + str(int(params.qr_run_n))+'/env_run_'+ str(int(params.env_run_n))
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
    return fig_dir    

# polynomials (fits)
def get_polynomials_fig_dir(params):
    fig_dir = 'fig/qr_run_' + str(int(params.qr_run_n))+'/env_run_'+ str(int(params.env_run_n))+'/poly_run_'+str(int(params.poly_run_n))
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
    return fig_dir    
