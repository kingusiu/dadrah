import pofah.jet_sample as jesa
import dadrah.util.run_paths as runpa
import dadrah.util.string_constants as stco
import dadrah.util.logging as log

# logging
logger = log.get_logger(__name__)


def read_sample(sample_id, params, filename):

    path_ext_dict = {'run': str(params.vae_run_n), filename: None}

    paths = runpa.RunPaths(in_data_dir=stco.dir_path_dict['base_dir_vae_results_qcd_train'], in_data_names=stco.file_name_path_dict, out_data_dir=stco.dir_path_dict['base_dir_qr_selections'])
    paths.extend_in_path_data(path_ext_dict)

    sample = jesa.JetSample.from_input_file(sample_id, paths.in_file_path(sample_id), read_n=params.read_n) 

    logger.info('read {} samples from {}'.format(len(sample), paths.in_file_path(sample_id)))

    return sample



