import json

import numpy as np


# https://stackoverflow.com/a/47626762/8670609
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)