####************************************************************************************************
###  Recursive Feature Elimination
###  We start will all the features & keep removing one feature in every iteration that provides a least value
###  Computation cost is high
###  For models like Logistic Reg / SVM the co-efficients tells us the feature importance
###  For tree based model we get feature importance in place of co-efficients
###  We remove the feature which has a co-efficient close to 0
###  the co-efficient are more positive if they are important for positive class & more negative
###  if they are more important to negative class
####************************************************************************************************

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
print(X.head())

model = LinearRegression()
rfe = RFE(estimator=model,n_features_to_select=3)
rfe.fit(X,y)
print(rfe.get_support(indices=False))
X_transformed = pd.DataFrame(rfe.transform(X),columns=X.columns[rfe.get_support(indices=True)])
print(X_transformed.head())

                    
                    
