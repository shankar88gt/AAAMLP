import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore")
import itertools

def Feature_engg(df,cat_cols):
    combi = list(itertools.combinations(cat_cols,2))
    for c1, c2 in combi:
        df.loc[:,c1+"_"+c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    return df


adult = fetch_ucirepo(id=2)
X = adult.data.features 
y = adult.data.targets 
df = pd.concat([X, y], axis=1)

num_cols = [
    'fnlwgt',
    'age',
    'capital-gain', 
    'capital-loss', 
    'hours-per-week'
]

target_mapping ={
        "<=50K" : 0,
        "<=50K." : 0,
        ">50K" : 1,
        ">50K." : 1
    }

df.loc[:,"income"] = df.income.map(target_mapping)

cat_cols = [
    c for c in df.columns if c not in num_cols and c not in ('kfold','income')
]

df = Feature_engg(df,cat_cols)

#Kfold
df['kfold'] = -1

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# extract target values
y = df.income.values

# stratified Kfolds in case target variable distribution is sckewed
kf= model_selection.StratifiedKFold(n_splits=5)
for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold'] = f

#Stats
print(df['kfold'].value_counts())
print(df.groupby(['kfold','income'])['income'].count())

# Save the Training file 
df.to_csv("/Users/shankarmanoharan/ML/Kaggle projects/Adult/Adult_Folds.csv",index=False)
print('Done')

def train(fold):

    df=pd.read_csv("/Users/shankarmanoharan/ML/Kaggle projects/Adult/Adult_Folds.csv")

    features = [ f for f in df.columns if f not in ['income','kfold']  ]

    lbl = preprocessing.LabelEncoder()
    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna("NONE")
            lbl.fit(df[col])
            df.loc[:,col] = lbl.transform(df[col])

    # training data is all but fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation is fold 
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Training data
    x_train = df_train[features].values

    # Validation Data
    x_valid = df_valid[features].values

    # Train the model on the Non fold mentioned & validation on Fold
    clf = xgb.XGBClassifier(n_jobs=-1,max_depth=7)
    clf.fit(x_train,df_train.income.values)

    preds = clf.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.income.values,preds)

    print(f"fold= {fold},AUC = {auc}")

for fold in range(5):
    train(fold) 




