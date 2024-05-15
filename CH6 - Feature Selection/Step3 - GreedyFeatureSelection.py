####************************************************************************************************
###  Greedy Feature Selection
###  It will fit a given model each time it evaluates a feature
###  Computation cost is high
###  Time required also very high 
###  If you do not use this correctly; you might overfit
###  The below approach will overfit as we have not split between training & Validation
###  The next version will implement this approach
####************************************************************************************************

import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

class GreedyFeatureSelection:
    def evaluate_score(self,X,y):
            # implment Kfold
            model= linear_model.Lasso(alpha=0.1)
            model.fit(X,y)            
            score = model.score(X,y)
            return score
    
    def FeatureSelection(self,X,y):
        ## This function does the actual greedy Selection
        ## initialize good features list  best scores to track both          
        good_features = []
        best_scores = []

        num_features = X.shape[1]
        print(num_features)
        
        this_feature = None
        best_score =0

        for feature in range(num_features):
            print(feature)
            if feature in good_features:
                continue
            selected_features = good_features + [feature] 
            x_train = X[:,selected_features]
            score = self.evaluate_score(x_train,y)
            print("Features" , selected_features)
            print("Score:" , score)
            if score > best_score:
                this_feature = feature
                best_score = score                
            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)
        return best_scores, good_features

    def __call__(self, *args: any, **kwds: any) -> any:
        print("inside Call")
        scores, features = self.FeatureSelection(X,Y)
        return X[:,features],scores

data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
df = pd.DataFrame(X,columns=col_names)
print(df.head())
Y = data["target"]
X_transformed,scores = GreedyFeatureSelection()(df,Y)
df1 = pd.DataFrame(X_transformed)
print(df1.columns)
print(scores)
                 
                
                    
                    
