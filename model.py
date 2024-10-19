from config import MODEL_PATH, LABEL_PATH
import pickle
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error


model = GradientBoostingClassifier()

iris_dataset = load_iris()
data, target, target_names = (
    iris_dataset["data"],
    iris_dataset["target"],
    iris_dataset["target_names"],
)

np.random.shuffle(data), np.random.shuffle(target)
train_x, train_y = data[:100], target[:100]
val_x, val_y = data[100:], target[100:]

model.fit(train_x, train_y)
print("MSE:", mean_squared_error(model.predict(val_x), val_y))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(LABEL_PATH, "w") as f:
    json.dump(target_names.tolist(), f)
    
    
