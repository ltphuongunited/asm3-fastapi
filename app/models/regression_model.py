import pickle
import numpy as np

with open("model/regression_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(input_data):
    data = np.array([[input_data.feature]])
    return model.predict(data).tolist()[0]
