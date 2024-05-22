import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from skopt import gp_minimize
from skopt import space

def optimize(params, param_names,x ,y ):
    """
    The main optimization funtion. this function takes all the arguments from the search space
    and training features & targets. it then initializes the model by setting the chosen parameters
    and runs cross validations & return negative accuracy
    For optimization we need a function the optimizer can minimize  usuallu the loss function
    we can also chose accuracy * -1 ; basically we are minimizing negative accuracy i.e maximizing accuracy
    I specifically prefer logictic regression loss function

    Returns negative accuracy after 5 folds

    Options to try
    1) can try logitic loss function
    2) accuracy * -1 can be replaced with 1-accuracy ( shd try )
    """

    # convert params to dictionary
    params = dict(zip(param_names,params))
    print(params)
    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)

    # initialize stratifiedK fold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # accuracy list
    accuracy = []

    #loop over all folds
    # In each fold the data is split into Test ( idx1 - 1 folds ) & train ( idx0 - 4 folds )
    # mean of 5 folds are calculated; total run time iteration i.e 15*5  
    # Check example Stratified K fold for understanding how this works
    for idx in kf.split(X=x,y=y):
        train_idx,test_idx = idx[0],idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain,ytrain)
        preds = model.predict(xtest)

        fold_accuracy = metrics.accuracy_score(ytest,preds)
        accuracy.append(fold_accuracy)
    return -1 * np.mean(accuracy)

df = pd.read_csv("/Users/shankarmanoharan/ML/Kaggle projects/MobilePricePred/train.csv")

X = df.drop('price_range',axis=1).values
y = df.price_range.values

param_names = ["max_depth","n_estimators","criterion","max_features"]
param_space = [
    space.Integer(3,15,name="max_depth"),
    space.Integer(100,1500,name="n_estimators"),
    space.Categorical(["gini","entropy"],name="criterion"),
    space.Real(0.01,1,prior="uniform",name="max_features")
]
optimization_funtion = partial(optimize,param_names = param_names,x=X,y=y)

result = gp_minimize(
    optimization_funtion,
    dimensions=param_space,
    n_calls = 15,
    n_random_starts=10,
    verbose =10
)

best_param = dict(zip(param_names,result.x) )

print(best_param)

# Convergence plot
from skopt.plots import plot_convergence
plot_convergence(result)