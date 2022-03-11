import numpy as np

import pofah.JetSample as jesa

def get_bin_counts_sig_like_bg_like(sample:jesa.JetSample, bin_edges:list) -> list[int]:
    
    tot_count, _ = np.histogram(sample['mJJ'], bins=bin_edges)
    acc_count, _ = np.histogram(sample.accepted('mJJ'), bins=bin_edges)
    rej_count, _ = np.histogram(sample.rejected('mJJ'), bins=bin_edges) 
    return [tot_count, acc_count, rej_count]


def divide_sample_into_orthogonal_quantiles(sample:jesa.JetSample, quantiles:list) -> list[jesa.JetSample]:

    pass