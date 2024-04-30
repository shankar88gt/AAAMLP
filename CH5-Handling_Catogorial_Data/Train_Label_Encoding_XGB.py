# src/train.py
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

import config
import warnings
warnings.filterwarnings("ignore")

import joblib
import os

def train():
    df=pd.read_csv(config.TRAINING_FILE_FOLDS)    

    features = [ f for f in df.columns if f not in ['id','target','kfold']  ]
    lbl = preprocessing.LabelEncoder()
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
        lbl.fit(df[col])
        df.loc[:,col] = lbl.transform(df[col])

    # training data is all but fold
    df_train = df[df.kfold != config.FOLD].reset_index(drop=True)
    # validation is fold 
    df_valid = df[df.kfold == config.FOLD].reset_index(drop=True)

    # Training data
    x_train = df_train[features].values
    y_train = df_train.target.values

    # Validation Data
    x_valid = df_valid[features].values
    y_valid = df_valid.target.values

    # Train the model on the Non fold mentioned & validation on Fold
    clf = xgb.XGBClassifier(n_estimators=200,n_jobs=-1,max_depth=7)
    clf.fit(x_train,y_train)

    preds = clf.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(y_valid,preds)
    
    print(f"fold= {config.FOLD},AUC = {auc}")

    #save the model
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f"dt_{config.FOLD}.bin"))

train()

