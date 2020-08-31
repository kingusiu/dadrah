import numpy as np

def get_bin_counts_sig_like_bg_like(sample, bin_edges):
    tot_count, _ = np.histogram(sample['mJJ'], bins=bin_edges)
    acc_count, _ = np.histogram(sample.accepted('mJJ'), bins=bin_edges)
    rej_count, _ = np.histogram(sample.rejected('mJJ'), bins=bin_edges) 
    return [tot_count, acc_count, rej_count]

