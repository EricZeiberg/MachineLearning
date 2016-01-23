import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("data/train.csv")

dataset.loc[dataset["target"] == "Class_1", "target"] = 1
dataset.loc[dataset["target"] == "Class_2", "target"] = 2
dataset.loc[dataset["target"] == "Class_3", "target"] = 3
dataset.loc[dataset["target"] == "Class_4", "target"] = 4
dataset.loc[dataset["target"] == "Class_5", "target"] = 5
dataset.loc[dataset["target"] == "Class_6", "target"] = 6
dataset.loc[dataset["target"] == "Class_7", "target"] = 7
dataset.loc[dataset["target"] == "Class_8", "target"] = 8
dataset.loc[dataset["target"] == "Class_9", "target"] = 9
dataset = dataset.fillna(0)

SVC = LinearSVC()
KNC = KNeighborsClassifier()
predictors = []
for x in range(1, 93):
    predictors.append('feat_' + str(x))

scores = cross_validation.cross_val_score(KNC, dataset[predictors], dataset["target"], cv=3)
print(scores.mean())
