import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

df  = pd.read_csv("/Users/shankarmanoharan/ML/Kaggle projects/MobilePricePred/train.csv")
print(df.shape)
print(df.head())

# seperate features & Target variable
X = df.drop('price_range',axis=1)
Y = df.price_range.values

#model
classifier = ensemble.RandomForestClassifier(n_jobs=-1)

#define Grid of Params
params = {
        "n_estimators" : np.arange(100,1500,100),
        "max_depth": np.arange(1,31),
        "criterion": ["gini","entropy"]
}

model = model_selection.RandomizedSearchCV(
    estimator=classifier,
    param_distributions=params,
    scoring='accuracy',
    n_iter=20,
    verbose=10,
    n_jobs=1,
    cv=5
)

model.fit(X,Y)
print('best Score: ', model.best_score_)

print("best parameters")
best_params = model.best_estimator_.get_params()
for param_name in sorted(params):
    print(param_name, best_params[param_name])

"""**********************************************************************************
[CV 5/5; 19/20] START criterion=entropy, max_depth=25, n_estimators=1000........
[CV 5/5; 19/20] END criterion=entropy, max_depth=25, n_estimators=1000;, score=0.870 total time=   1.3s
[CV 1/5; 20/20] START criterion=entropy, max_depth=11, n_estimators=100.........
[CV 1/5; 20/20] END criterion=entropy, max_depth=11, n_estimators=100;, score=0.873 total time=   0.2s
[CV 2/5; 20/20] START criterion=entropy, max_depth=11, n_estimators=100.........
[CV 2/5; 20/20] END criterion=entropy, max_depth=11, n_estimators=100;, score=0.880 total time=   0.2s
[CV 3/5; 20/20] START criterion=entropy, max_depth=11, n_estimators=100.........
[CV 3/5; 20/20] END criterion=entropy, max_depth=11, n_estimators=100;, score=0.902 total time=   0.2s
[CV 4/5; 20/20] START criterion=entropy, max_depth=11, n_estimators=100.........
[CV 4/5; 20/20] END criterion=entropy, max_depth=11, n_estimators=100;, score=0.865 total time=   0.1s
[CV 5/5; 20/20] START criterion=entropy, max_depth=11, n_estimators=100.........
[CV 5/5; 20/20] END criterion=entropy, max_depth=11, n_estimators=100;, score=0.845 total time=   0.1s
best Score:  0.8879999999999999
best parameters
criterion entropy
max_depth 13
n_estimators 200
*************************************************************************************"""