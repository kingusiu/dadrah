import os
import pofah.jet_sample as js

def merge_qcd_base_and_ext_datasets(params, paths, **cuts):
    # read qcd & qcd ext
    qcd_sr_sample = js.JetSample.from_input_dir(params.qcd_sample_id, paths.sample_dir_path(params.qcd_sample_id), read_n=params.read_n, **cuts) 
    qcd_sr_ext_sample = js.JetSample.from_input_dir(params.qcd_ext_sample_id, paths.sample_dir_path(params.qcd_ext_sample_id), read_n=params.read_n, **cuts)
    # merge to combined jet sample and split into training and test parts
    return qcd_sr_sample.merge(qcd_sr_ext_sample) 


def make_qcd_train_test_datasets(params, paths, train_split=0.2, **cuts):
    # merge to combined jet sample and split into training and test parts
    qcd_sr_all_sample = merge_qcd_base_and_ext_datasets(params, paths, **cuts) 
    qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sr_all_sample, train_split, new_names=(params.qcd_train_sample_id, params.qcd_test_sample_id)) # can train on max 4M events (20% of qcd SR)
    # write to file
    qcd_train.dump(paths.sample_file_path(params.qcd_train_sample_id, mkdir=True))
    qcd_test.dump(paths.sample_file_path(params.qcd_test_sample_id, mkdir=True))
    return qcd_train, qcd_test


def inject_signal(qcd_train_sample, sig_sample, sig_in_training_num, train_vs_valid_split=0.8):
    if sig_in_training_num == 0:
        return js.split_jet_sample_train_test(qcd_train_sample, train_vs_valid_split)
    # sample random sig_in_train_num events from signal sample
    sig_train_sample = sig_sample.sample(n=sig_in_training_num)
    # merge qcd and signal
    mixed_sample = qcd_train_sample.merge(sig_train_sample)
    # split training data into train and validation set
    mixed_sample_train, mixed_sample_valid = js.split_jet_sample_train_test(mixed_sample, train_vs_valid_split)
    return mixed_sample_train, mixed_sample_valid

