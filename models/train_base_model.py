# Import Libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, brier_score_loss

#import dataset, convert to dataframe, print first 10 rows:
raw_data_path = "data/raw/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(raw_data_path)
#print(df.head(10))

#split dataset into features and target varibles:
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

#split dataset into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

#define features to be scaled an imputed: 
numeric_features = X.columns.tolist()

#define preprocessing steps for features:
preprocessor = ColumnTransformer(
    transformers=[
            ('num', Pipeline(
                [('imputer', SimpleImputer(strategy='median')),
                 ('scaler', StandardScaler())
                ]), numeric_features)
    ]
)

