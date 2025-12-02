import pickle
import numpy as np

# load model
with open("knn_model.sav", "rb") as f:
    clf = pickle.load(f)

labels = ["setosa", "versicolor", "virginica"]

def predict(data):
    pred_idx = clf.predict(data)[0]
    return labels[pred_idx]
