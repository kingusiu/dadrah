import datetime
from enum import Enum

def make_qr_model_str(run_n_qr, run_n_vae, quantile, sig_id, sig_xsec, strategy_id):
    return 'QRmodel_run_{}_vae_run_{}_qnt_{}_{}_sigx_{}_loss_{}.h5'.format(run_n_qr, run_n_vae, str(int(quantile*100)), sig_id, int(sig_xsec), strategy_id)

def quantile_str(quantile):
    return 'q{:02}'.format(int(quantile*100))

def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return quantile_str(inv_quant)




dir_path_dict = {
    
    'base_dir_vae_results_qcd_train' : '/eos/user/k/kiwoznia/data/VAE_results/events/',
    'base_dir_qr_selections' : '/eos/user/k/kiwoznia/data/QR_results/events/',
    'base_dir_qr_selections_poly_cut' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/QR_models/envelope/fitted_selections/',
    'base_dir_qr_analysis' : 'fig', #'/eos/user/k/kiwoznia/data/QR_results/analysis/'
}

file_name_path_dict = {
    
    'qcdSigAllTestReco' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_reco.h5',
    'qcdSigAllTrain30pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Train_30.h5',
    'qcdSigAllTest30pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_30.h5',
    'qcdSigAllTest50pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_50.h5',
    'qcdSigAllTest70pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_70.h5',
    'qcdSigAllTest80pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_80.h5',
    'GtoWW35naReco' : 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_reco.h5'
}

# 3 QR model option: regular dense QR, polynomial fit of 3rd degree, bernstein polynomial fit of 3rd degree
QR_Model = Enum('QR_Model', 'DENSE POLY BERNSTEIN')

