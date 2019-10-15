import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

#Function to clean erroneous data in the dataset
def cleanData(dataframe):
    dataframe["Gender"] = dataframe["Gender"].replace(["unknown", "0"], np.nan)
    dataframe["University Degree"] = dataframe["University Degree"].replace("0", "No")
    dataframe["Hair Color"] = dataframe["Hair Color"].replace(["Unknown","0"], np.nan)
    if "Income in EUR" in dataframe.columns:
        dataframe.loc[dataframe["Income in EUR"] < 0, "Income in EUR"] = np.nan
    else:
        pass

#Function to handle all the NaN values in the dataset
def handleNaNs(dataframe):
    dataframe["Gender"].fillna(dataframe["Gender"].mode()[0], inplace = True)
    dataframe["Age"].fillna(dataframe.groupby("Gender")["Age"].transform("mean"), inplace = True)
    dataframe["Body Height [cm]"].fillna(dataframe.groupby("Gender")["Body Height [cm]"].transform("mean"), inplace = True)
    dataframe["Hair Color"].fillna(dataframe["Hair Color"].mode()[0], inplace = True)
    dataframe["University Degree"].fillna(dataframe["University Degree"].mode()[0], inplace = True)
    dataframe["Country"].fillna("Unknown", inplace = True)
    dataframe["Size of City"].fillna(dataframe.groupby("Country")["Size of City"].transform("mean"), inplace = True)
    dataframe["Profession"].fillna("Unknown", inplace = True)
    dataframe["Year of Record"].fillna(method = "ffill", inplace = True)
    if "Income in EUR" in dataframe.columns:
        dataframe["Income in EUR"].fillna(dataframe.groupby("Year of Record")["Income in EUR"].transform("mean"), inplace = True)
    else:
        pass

#Function to add noise. Used in the target_encode() function below.
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

# Function to perform target encoding.
# Used code that I found online to perform target encoding on the categorical features.
# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply aver
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

#Function to call the target_encode() function to encode all the categorical columns properly
def encodeDataframe(xTrain, xTest, yTrain):
    xTrain["Gender"], xTest["Gender"] = target_encode(xTrain["Gender"], xTest["Gender"],yTrain)
    xTrain["Country"], xTest["Country"] = target_encode(xTrain["Country"], xTest["Country"],yTrain)
    xTrain["Profession"], xTest["Profession"] = target_encode(xTrain["Profession"], xTest["Profession"],yTrain)
    xTrain["University Degree"], xTest["University Degree"] = target_encode(xTrain["University Degree"], xTest["University Degree"],yTrain)
    xTrain["Hair Color"], xTest["Hair Color"] = target_encode(xTrain["Hair Color"], xTest["Hair Color"], yTrain)

#Function to use gradient boosting regressor to predict the values.
def predictor(xTrain, xTest, yTrain):
    #Create feature transformation and training pipeline
    gbm = GradientBoostingRegressor(learning_rate = 0.1, random_state = 1234)
    pipe = Pipeline([("gbm", gbm)])

    #Fit model
    gbm_cv = GridSearchCV(pipe, dict(gbm__n_estimators = [50, 100, 150], gbm__max_depth = [5, 6, 7]), cv = 5, scoring = make_scorer(mean_squared_error), verbose = 100)
    gbm_cv.fit(xTrain, yTrain)
    
    #Predictor
    yPred = gbm_cv.best_estimator_.predict(xTest)
    
    return yPred

#Function used to quickly test the efficeny of the model.
def tester(yTest, yPred):
    #Mean Squared Error
    mse = mean_squared_error(yTest, yPred)
    print("Mean squared error: %.2f" % mse)

    # The Root Mean Squared Error (RMSE)
    print("Root Mean Squared Error: %.2f" % sqrt(mse))

    # Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % r2_score(yTest, yPred))

#Importing the training dataset
df = pd.read_csv("/Users/kavith/Desktop/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")

#Dropping the Instance column as it is not required.
df = df.drop(["Instance"], axis = 1)

#Calling the cleanData() and handleNaNs() functions to preprocess the dataset.
cleanData(df)
handleNaNs(df)

#Removing the Income in EUR column from X and making it separate.
Y = df["Income in EUR"]
X = df
X = X.drop(["Income in EUR"], axis = 1)

#Splitting the training dataset in a 70-30 ratio to test the model
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)

#Calling the encodeDataframe() function to target encode the categorical features present in the dataset.
encodeDataframe(x_train, x_test, y_train)

#Predicting the values in the test part of the training dataset.
y_pred = predictor(x_train, x_test, y_train)

#Testing the model to see how it performs. 
tester(y_test, y_pred)

#Importing the test dataset and preprocessing it.
testdf = pd.read_csv("/Users/kavith/Desktop/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv")
testdf = testdf.drop(["Income", "Instance"], axis = 1)
cleanData(testdf)
handleNaNs(testdf)

#Calling the encodeDataframe() function to target encode the categorical features present in the test dataset.
encodeDataframe(X, testdf, Y)

#Predicting the income in the test dataset
y_pred = predictor(X, testdf, Y)

#Saving the output in aa csv file
np.savetxt("out_gbr.csv", y_pred)