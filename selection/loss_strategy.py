from collections import OrderedDict
import numpy as np

'''
    5 loss combination strategies:
    - L1 > LT
    - L2 > LT
    - L1 + L2 > LT
    - L1 | L2 > LT
    - L1 & L2 > LT
    x ... jet_sample (pofah) or pandas data-frame (or rec-array)
'''

def combine_loss_l1(x):
    """ L1 > LT """
    return x['j1TotalLoss']

def combine_loss_l2(x):
    """ L2 > LT """
    return x['j2TotalLoss']

def combine_loss_sum(x):
    """ L1 + L2 > LT """
    return x['j1TotalLoss'] + x['j2TotalLoss']

def combine_loss_max(x):
    """ L1 | L2 > LT """
    return np.maximum(x['j1TotalLoss'], x['j2TotalLoss'])

def combine_loss_min(x):
    """ L1 & L2 > LT """
    return np.minimum(x['j1TotalLoss'], x['j2TotalLoss'])

def combine_loss_reco_min(x):
    """ Reco L1 & Reco L2 > LT """
    return np.minimum(x['j1RecoLoss'], x['j2RecoLoss'])

def combine_loss_r1(x):
    """ L1 > LT """
    return x['j1RecoLoss']

def combine_loss_r2(x):
    """ L2 > LT """
    return x['j2RecoLoss']

def combine_loss_kl1(x):
    ''' KL J1 '''
    return x['j1KlLoss']

def combine_loss_kl2(x):
    ''' KL J2 '''
    return x['j2KlLoss']

def combine_loss_kl_sum(x):
    ''' KL J1 + KL J2 '''
    return x['j1KlLoss'] + x['j2KlLoss']

def combine_loss_kl_max(x):
    ''' KL J1 | KL J2 '''
    return np.maximum(x['j1KlLoss'], x['j2KlLoss'])

def combine_loss_kl_min(x):
    ''' KL J1 & KL J2 '''
    return np.minimum(x['j1KlLoss'], x['j2KlLoss'])

def combine_loss_reco_kl_min(x, beta=10.):
    ''' (RecoLoss J1 + 10* KL J1) & (RecoLoss J2 + 10* KL J2) '''
    return np.minimum(x['j1RecoLoss'] + beta * x['j1KlLoss'], x['j2RecoLoss'] + beta * x['j2KlLoss']) 



class LossStrategy():

    def __init__(self, loss_fun, title_str, file_str):
        self.fun = loss_fun
        self.title_str = title_str
        self.file_str = file_str

    def __call__(self, x):
        return self.fun(x)


class LossStrategyParam(LossStrategy):
    ''' parametrized loss strategy '''
    def __init__(self, param, **kwargs):
        super().__init__(**kwargs)
        self.param = param

    def __call__(self, x):
        return self.fun(x, self.param)


loss_strategy_dict = OrderedDict({ 
                     's1' : LossStrategy(loss_fun=combine_loss_l1, title_str='L1 > LT', file_str='l1_loss'),
                     's2': LossStrategy(loss_fun=combine_loss_l2, title_str='L2 > LT', file_str='l2_loss'),
                     's3': LossStrategy(loss_fun=combine_loss_sum, title_str='L1 + L2 > LT', file_str='suml1l2_loss'),
                     's4': LossStrategy(loss_fun=combine_loss_max, title_str='L1 | L2 > LT', file_str='maxl1l2_loss'),
                     's5': LossStrategy(loss_fun=combine_loss_min, title_str='L1 & L2 > LT', file_str='minl1l2_loss'),
                     'r1' : LossStrategy(loss_fun=combine_loss_r1, title_str='R1 > LT', file_str='r1_loss'),
                     'r2' : LossStrategy(loss_fun=combine_loss_r2, title_str='R2 > LT', file_str='r2_loss'),
                     'r5': LossStrategy(loss_fun=combine_loss_reco_min, title_str='R1 & R2 > LT', file_str='min_reco1reco2_loss'),
                     'kl1': LossStrategy(loss_fun=combine_loss_kl1, title_str='KL J1 > LT', file_str='kl1_loss'),
                     'kl2': LossStrategy(loss_fun=combine_loss_kl2, title_str='KL J2 > LT', file_str='kl2_loss'),
                     'kl3': LossStrategy(loss_fun=combine_loss_kl_sum, title_str='KL J1 + KL J2 > LT', file_str='sumKL_loss'),
                     'kl4': LossStrategy(loss_fun=combine_loss_kl_max, title_str='KL J1 | KL J2 > LT', file_str='maxKL_loss'),
                     'kl5': LossStrategy(loss_fun=combine_loss_kl_min, title_str='KL J1 & KL J2 > LT', file_str='minKL_loss'),
                     'rk5_10': LossStrategyParam(param=10., loss_fun=combine_loss_reco_kl_min, title_str='(R J1 + 10*KL J1) & (R J2 + 10*KL J2)', file_str='min_recoKLb10_loss'), 
                     'rk5_1': LossStrategyParam(param=1., loss_fun=combine_loss_reco_kl_min, title_str='(R J1 + KL J1) & (R J2 + KL J2)', file_str='min_recoKLb1_loss'), 
                     'rk5_01': LossStrategyParam(param=0.1, loss_fun=combine_loss_reco_kl_min, title_str='(R J1 + 0.1*KL J1) & (R J2 + 0.1*KL J2)', file_str='min_recoKLb01_loss'), 
                     'rk5_05': LossStrategyParam(param=0.5, loss_fun=combine_loss_reco_kl_min, title_str='(R J1 + 0.5*KL J1) & (R J2 + 0.5*KL J2)', file_str='min_recoKLb05_loss'), 
                 })
