import dadrah.selection.quantile_regression as qure
import numpy as np


# *************************************
#				data
# *************************************



# *************************************
#				model
# *************************************

QR = qure.QuantileRegressionV2(n_layers=2, n_nodes=10).make_model()