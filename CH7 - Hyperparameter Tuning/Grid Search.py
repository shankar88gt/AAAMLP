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
        "n_estimators" : [100,200,300,400,500],
        "max_depth": [1,5,7,9,11,15,20],
        "criterion": ["gini","entropy"]
}

model = model_selection.GridSearchCV(
    estimator=classifier,
    param_grid=params,
    scoring='accuracy',
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
Sample Output
[CV 2/5; 70/70] END criterion=entropy, max_depth=20, n_estimators=500;, score=0.892 total time=   0.9s
[CV 3/5; 70/70] START criterion=entropy, max_depth=20, n_estimators=500.........
[CV 3/5; 70/70] END criterion=entropy, max_depth=20, n_estimators=500;, score=0.912 total time=   0.8s
[CV 4/5; 70/70] START criterion=entropy, max_depth=20, n_estimators=500.........
[CV 4/5; 70/70] END criterion=entropy, max_depth=20, n_estimators=500;, score=0.875 total time=   0.8s
[CV 5/5; 70/70] START criterion=entropy, max_depth=20, n_estimators=500.........
[CV 5/5; 70/70] END criterion=entropy, max_depth=20, n_estimators=500;, score=0.865 total time=   0.8s
best Score:  0.8869999999999999
best parameters
criterion entropy
max_depth 15
n_estimators 300
*************************************************************************************"""