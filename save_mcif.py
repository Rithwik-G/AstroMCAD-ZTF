from sklearn.ensemble import IsolationForest
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
import os

export_path = "exports/iforests"

class mcif:
    def __init__(self, n_estimators = 100):
        self.n_estimators=n_estimators
    
    def train(self, x_data, labels):

        self.classes = np.unique(labels)
        self.iforests = [IsolationForest(n_estimators=self.n_estimators) for i in self.classes]
        
        for ind, class_name in enumerate(self.classes):
            here = []
            for i in range(len(x_data)):
                if (class_name == labels[i]):
                    here.append(x_data[i])

            self.iforests[ind].fit(here)

    def score(self, data):
        return [np.min(i) for i in self.score_discrete(data)]

    def score_discrete(self, data):
        scores = []
        for ind, forest in enumerate(self.iforests):
            scores.append(forest.decision_function(data))
        return scores


model_path = '/jet/home/rgupta9/A Organized Home/Anomaly Detection/astromcad/Models/NoHostMCIF_new.pickle'

with open(model_path, "rb") as f:
    model = pickle.load(f)


onnx_models = {}

for cls, forest in zip(model.classes, model.iforests):
    initial_type = [('input', FloatTensorType([None, 100]))]
    onnx_model = convert_sklearn(
        forest,
        initial_types=initial_type,
        target_opset={'': 13, 'ai.onnx.ml': 3}
    )
    onnx_models[cls] = onnx_model

os.makedirs(export_path, exist_ok=True)

for cls, onnx_model in onnx_models.items():
    with open(f"{export_path}/iforest_{cls}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

