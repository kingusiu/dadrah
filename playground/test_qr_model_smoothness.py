import dadrah.util.run_paths as runpa
import dadrah.selection.loss_strategy as lost




if __name__ == '__main__':

    vae_run_n = 113
    qr_run_n = 111
    sample_ids = ['qcdSigAllTestReco', 'GtoWW35naReco']
    quantiles = [0.3, 0.5, 0.7, 0.9]
    # quantiles = [0.9]
    sig_xsec = 0.
    loss_strategy = lost.loss_strategy_dict['rk5_05']
    in_path_ext_dict = {}
    path_ext_dict = {'vae_run': str(vae_run_n), 'qr_run': str(qr_run_n), 'sig': sample_ids[1], 'xsec': str(int(sig_xsec)), 'loss': 'rk5_05'}

    ### paths ###

    # data inputs (mjj & vae-scores): /eos/user/k/kiwoznia/data/VAE_results/events/run_$vae_run_n$
    # data outputs (selections [0/1] per quantile): /eos/user/k/kiwoznia/data/QR_results/events/

    paths = runpa.RunPaths(in_data_dir=strc.dir_path_dict['base_dir_vae_results'], in_data_names=strc.file_name_path_dict, out_data_dir=strc.dir_path_dict['base_dir_qr_selections'])
    paths.extend_in_path_data(path_ext_dict)
    paths.extend_out_path_data({**path_ext_dict, 'ortho_quantiles': None})

    print('plotting figures to ' + paths.out_data_dir)