import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import Lasso

training_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv('data/test.csv')
stores = pd.read_csv("data/store.csv")


training_data['Date'] = pd.to_datetime(training_data['Date'])
training_data['Date'] = training_data['Date'].astype(np.int64) // 10**9

dataset = pd.concat([training_data, stores])

dataset["Promo2"] = dataset.Promo2.fillna(0)
dataset["Promo2SinceYear"] = dataset.Promo2SinceYear.fillna(dataset['Promo2SinceYear'].median())
dataset["PromoInterval"] = dataset.PromoInterval.fillna(0)
dataset["Open"] = dataset.Open.fillna(1)


dataset["CompetitionOpenSinceYear"] = dataset["CompetitionOpenSinceYear"].fillna(dataset["CompetitionOpenSinceYear"].median())
dataset["Customers"] = dataset["Customers"].fillna(dataset["Customers"].median())

dataset["StoreType"] = dataset["StoreType"].fillna(dataset["StoreType"].mode())
dataset["Assortment"] = dataset["Assortment"].fillna(dataset["Assortment"].mode())


dataset.loc[dataset["StoreType"] == "a", "StoreType"] = 0
dataset.loc[dataset["StoreType"] == "b", "StoreType"] = 1
dataset.loc[dataset["StoreType"] == "c", "StoreType"] = 2
dataset.loc[dataset["StoreType"] == "d", "StoreType"] = 3
dataset.loc[dataset["Assortment"] == "a", "Assortment"] = 0
dataset.loc[dataset["Assortment"] == "b", "Assortment"] = 1
dataset.loc[dataset["Assortment"] == "c", "Assortment"] = 2
dataset.loc[dataset["Assortment"] == "d", "Assortment"] = 3
dataset = dataset.fillna(0)

# Now do the same with test_data
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['Date'] = test_data['Date'].astype(np.int64) // 10**9

merged_test_data = pd.concat([test_data, stores])


merged_test_data["Promo2"] = merged_test_data.Promo2.fillna(0)
merged_test_data["Promo2SinceYear"] = merged_test_data.Promo2SinceYear.fillna(merged_test_data['Promo2SinceYear'].median())
merged_test_data["PromoInterval"] = merged_test_data.PromoInterval.fillna(0)
merged_test_data["Open"] = merged_test_data.Open.fillna(1)


merged_test_data["CompetitionOpenSinceYear"] = merged_test_data["CompetitionOpenSinceYear"].fillna(merged_test_data["CompetitionOpenSinceYear"].median())

merged_test_data["StoreType"] = merged_test_data["StoreType"].fillna(merged_test_data["StoreType"].mode())
merged_test_data["Assortment"] = merged_test_data["Assortment"].fillna(merged_test_data["Assortment"].mode())

merged_test_data.loc[merged_test_data["StoreType"] == "a", "StoreType"] = 0
merged_test_data.loc[merged_test_data["StoreType"] == "b", "StoreType"] = 1
merged_test_data.loc[merged_test_data["StoreType"] == "c", "StoreType"] = 2
merged_test_data.loc[merged_test_data["StoreType"] == "d", "StoreType"] = 3
merged_test_data.loc[merged_test_data["Assortment"] == "a", "Assortment"] = 0
merged_test_data.loc[merged_test_data["Assortment"] == "b", "Assortment"] = 1
merged_test_data.loc[merged_test_data["Assortment"] == "c", "Assortment"] = 2
merged_test_data.loc[merged_test_data["Assortment"] == "d", "Assortment"] = 3
merged_test_data = merged_test_data.fillna(0)

las = Lasso()
predictors = ['DayOfWeek', 'Date', 'Promo', 'Promo2', 'Promo2SinceYear', 'Assortment', 'StoreType', 'CompetitionDistance']
las.fit(dataset[predictors], dataset["Sales"])
merged_test_data = merged_test_data[merged_test_data.Id != 0]
predictions = las.predict(merged_test_data[predictors])
submission = pd.DataFrame({
        "Id": merged_test_data["Id"].astype(int),
        "Sales": predictions
    })
submission = submission[submission.Id != 0]
submission.to_csv("kaggle.csv", index=False)
#scores = cross_validation.cross_val_score(las, dataset[predictors], dataset["Sales"], cv=3)
#print(scores.mean())
