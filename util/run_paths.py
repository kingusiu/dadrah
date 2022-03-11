import os
import pathlib


class RunPaths():

    """
        Collection of input and output paths used in the run of a generic experiment 
        (training, inference, analysis) for data and/or models.
    """

    def __init__(self, in_data_dir:str=None, in_data_names:dict=None, in_model_dir:str=None, in_model_names:dict=None, 
                    out_data_dir:str=None, out_data_names:dict=None, out_model_dir:str=None, out_model_names:dict=None):

        self.path_dict = {

            'in' : {
                'data': {
                    'dir' : in_data_dir, # directory path
                    'names' : in_data_names, # dict with filenames
                },
                'model': {
                    'dir' : in_model_dir, # directory path
                    'names' : in_model_names, # dict with modelnames
                },
            }

            'out': {
                'data': {
                    'dir' : out_data_dir,
                    'names' : out_data_names,
                },
                'model': {
                    'dir' : out_model_dir,
                    'names' : out_model_names,
                },
            }
        }

        for inout in self.path_dict:
            for datmod in inout:
                if datmod['dir'] is not None:
                    pathlib.Path(datmod['dir']).mkdir(parents=True, exist_ok=True)


    @property
    def in_data_dir(self):
        return self.path_dict['in']['data']['dir']


    @property
    def out_data_dir(self):
        return self.path_dict['out']['data']['dir']


    def extend_path(self, path:str, param_dict:dict) -> str:

        for k, v in param_dict.items():
            path = os.path.join(path, k+'_'+v if v is not None else k)

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        return path


    def extend_path_data(self, param_dict:dict) -> None:

        self.path_dict['in']['data']['dir'] = self.extend_path(self.path_dict['in']['data']['dir'], param_dict)
        self.path_dict['out']['data']['dir'] = self.extend_path(self.path_dict['out']['data']['dir'], param_dict)


    def extend_in_path_data(self, param_dict:dict) -> None:

        self.path_dict['in']['data']['dir'] = self.extend_path(self.path_dict['in']['data']['dir'], param_dict)


    def extend_in_path_model(self, param_dict:dict) -> None:

        self.path_dict['in']['model']['dir'] = self.extend_path(self.path_dict['in']['model']['dir'], param_dict)


    def extend_out_path_data(self, param_dict:dict) -> None:

        self.path_dict['out']['data']['dir'] = self.extend_path(self.path_dict['out']['data']['dir'], param_dict)


    def extend_out_path_model(self, param_dict:dict) -> None:

        self.path_dict['out']['model']['dir'] = self.extend_path(self.path_dict['out']['model']['dir'], param_dict)


    def in_file_path(self, id:str) -> str:
        
        return os.path.join(self.path_dict['in']['data']['dir'], self.path_dict['in']['data']['names'][id])

    def out_file_path(self, id:str) -> str:
        
        return os.path.join(self.path_dict['out']['data']['dir'], self.path_dict['out']['data']['names'][id])


