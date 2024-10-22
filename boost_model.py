from ray import serve

import pickle
import json
import os
from starlette.requests import Request
from typing import Dict

# Save the model and label to file
MODEL_PATH = os.path.join("./iris_model_gradient_boosting_classifier.pkl")
LABEL_PATH = os.path.join("./iris_labels.json")

    
@serve.deployment
class BoostingModel:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(LABEL_PATH) as f:
            self.label_list = json.load(f)

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print("Worker: received starlette request with data", payload)

        input_vector = [
            payload["sepal length"],
            payload["sepal width"],
            payload["petal length"],
            payload["petal width"],
        ]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}
    
boosting_model = BoostingModel.bind()
