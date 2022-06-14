import datetime
from enum import Enum

def make_qr_model_str(run_n_qr, run_n_vae, quantile, sig_id, sig_xsec, strategy_id, date=None):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    return 'QRmodel_run_{}_vae_run_{}_qnt_{}_{}_sigx_{}_loss_{}_{}.h5'.format(run_n_qr, run_n_vae, str(int(quantile*100)), sig_id, int(sig_xsec), strategy_id, date)


def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))


dir_path_dict = {
    
    'base_dir_vae_results' : '/eos/user/k/kiwoznia/data/VAE_results/events/',
    'base_dir_qr_selections' : '/eos/user/k/kiwoznia/data/QR_results/events/',
    'base_dir_qr_selections_poly_cut' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/QR_models/envelope/fitted_selections/',
    'base_dir_qr_analysis' : 'fig', #'/eos/user/k/kiwoznia/data/QR_results/analysis/'
}

file_name_path_dict = {
    
    'qcdSigAllTestReco' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_reco.h5',
    'qcdSigAllTest30pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_30.h5',
    'qcdSigAllTest50pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_50.h5',
    'qcdSigAllTest80pct' : 'qcd_sqrtshatTeV_13TeV_PU40_NEW_ALL_Test_80.h5',
    'GtoWW35naReco' : 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_reco.h5'
}

# 3 QR model option: regular dense QR, polynomial fit of 3rd degree, bernstein polynomial fit of 3rd degree
QR_Model = Enum('QR_Model', 'DENSE POLY BERNSTEIN')

