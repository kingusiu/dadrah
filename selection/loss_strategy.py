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

def combine_loss_l1( x ):
    """ L1 > LT """
    return x['j1TotalLoss']

def combine_loss_l2( x ):
    """ L2 > LT """
    return x['j2TotalLoss']

def combine_loss_sum( x ):
    """ L1 + L2 > LT """
    return x['j1TotalLoss'] + x['j2TotalLoss']

def combine_loss_max( x ):
    """ L1 | L2 > LT """
    return np.maximum(x['j1TotalLoss'],x['j2TotalLoss'])

def combine_loss_min( x ):
    """ L1 & L2 > LT """
    return np.minimum(x['j1TotalLoss'],x['j2TotalLoss'])
