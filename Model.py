# Created by Myeisha Foo u3241507
import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass

# Data file import
df = pd.read_csv("surveylungcancer.csv")

# Attribute to be predicted
predict = "LUNG_CANCER"

# pre-processing
# encode object columns to integers
for col in df:
    if df[col].dtype == 'object':
        df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))

# Dataset to be Predicted, X is all attributes and y is the features
# Create a label encoder


le = preprocessing.LabelEncoder()

# Encode the new variables
GENDER = le.fit_transform(list(df["GENDER"]))
AGE = le.fit_transform(list(df["AGE"]))
SMOKING = le.fit_transform(list(df["SMOKING"]))
YELLOW_FINGERS = le.fit_transform(list(df["YELLOW_FINGERS"]))
ANXIETY = le.fit_transform(list(df["ANXIETY"]))
PEER_PRESSURE = le.fit_transform(list(df["PEER_PRESSURE"]))
CHRONIC_DISEASE = le.fit_transform(list(df["CHRONIC DISEASE"]))
FATIGUE = le.fit_transform(list(df["FATIGUE "]))  # Note the space in column name
ALLERGY = le.fit_transform(list(df["ALLERGY "]))  # Note the space in column name
WHEEZING = le.fit_transform(list(df["WHEEZING"]))
ALCOHOL_CONSUMING = le.fit_transform(list(df["ALCOHOL CONSUMING"]))
COUGHING = le.fit_transform(list(df["COUGHING"]))
SHORTNESS_OF_BREATH = le.fit_transform(list(df["SHORTNESS OF BREATH"]))
SWALLOWING_DIFFICULTY = le.fit_transform(list(df["SWALLOWING DIFFICULTY"]))
CHEST_PAIN = le.fit_transform(list(df["CHEST PAIN"]))
LUNG_CANCER = le.fit_transform(list(df["LUNG_CANCER"]))  # Note the column name change

# Model Test/Train
# Test options and evaluation metric
# Create a list of tuples, where each tuple contains values from different variables
x = list(zip(GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN ))

# Create a list for the target variable
y = list(LUNG_CANCER)

# Define options and evaluation metric for model testing
num_folds = 5  # Number of cross-validation folds
seed = 7  # Random seed for reproducibility
scoring = 'accuracy'  # Metric to evaluate model performance (accuracy in this case)


# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and similarly for Y(features)
# The test data will test the accuracy of the model created
import sklearn.model_selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
# 0.2 means 80% training 20% testing
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows

# Prediction class for import
@dataclass(eq=True, frozen=True, order=True)
class Prediction:
    best_model = RandomForestClassifier()
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    model_accuracy = accuracy_score(y_test, y_pred)