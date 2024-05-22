import numpy as np
from sklearn import model_selection
from sklearn import ensemble
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1,10], [12,78], [56,89], [12,56],[ 67,79], [34,79], [34,70]])
y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
kf = model_selection.StratifiedKFold(n_splits=5)
for idx in kf.split(X,y):
    print(idx)
    train_idx,test_idx = idx[0],idx[1]
    xtrain = X[train_idx]
    ytrain = y[train_idx]

    xtest = X[test_idx]
    ytest = y[test_idx]
    print(xtrain)
    print(xtest) 
