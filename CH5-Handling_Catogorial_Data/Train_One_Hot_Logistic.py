# src/train.py
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn import linear_model

import config
import model_dispatcher
import warnings
warnings.filterwarnings("ignore")

import joblib
import os

def train():
    df=pd.read_csv(config.TRAINING_FILE_FOLDS)    

    features = [ f for f in df.columns if f not in ['id','target','kfold']  ]
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")

    # training data is all but fold
    df_train = df[df.kfold != config.FOLD].reset_index(drop=True)

    # validation is fold 
    df_valid = df[df.kfold == config.FOLD].reset_index(drop=True)

    # Apply one hot encoding on both train + validation data
    one = preprocessing.OneHotEncoder()
    full_data = pd.concat([df_train[features],df_valid[features]],axis=0)
    one.fit(full_data[features])

    # Training data
    x_train = one.transform(df_train[features])
    y_train = df_train.target.values

    # Validation Data
    x_valid = one.transform(df_valid[features])
    y_valid = df_valid.target.values

    # Train the model on the Non fold mentioned & validation on Fold
    clf = linear_model.LogisticRegression()
    clf.fit(x_train,y_train)

    preds = clf.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(y_valid,preds)

    print(f"fold= {config.FOLD},AUC = {auc}")

    #save the model
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f"dt_{config.FOLD}.bin"))

train()

