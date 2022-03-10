
class RunPaths():

    """
        Collection of input and output paths used in the run of a generic experiment 
        (training, inference, analysis) for data and/or models.
    """


    def __init__(self, input_path_data:str, input_path_model:str=None, output_path_data:str=None, output_path_model:str=None):

        self.input_path_data = input_path_data
        self.input_path_model = input_path_model
        self.output_path_data = output_path_data
        self.output_path_model = output_path_model


    

