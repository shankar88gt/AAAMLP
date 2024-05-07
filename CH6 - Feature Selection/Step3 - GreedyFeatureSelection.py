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
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    def evaluate_score(self,X,y):
            # implment Kfold
            model= linear_model.logisticRegression()
            model.fit(X,y)
            preds = model.predict_prob(X)[:,1]
            auc = metrics.roc_auc_score(y,preds)
            return auc
    
    def FeatureSelection(self,X,y):
        ## This function does the actual greedy Selection
        ## initialize good features list  best scores to track both          
        good_features = []
        best_scores = []

        num_features = X.shape[1]

        while True:
             this_feature = None
             best_score =0

             for feature in range(num_features):
                if feature in good_features:
                    continue
                selected_features = good_features + [feature] 
                x_train = X[:,selected_features]
                score = self.evaluate_score(x_train,y)
                if score > best_score:
                  this_feature = feature
                  best_score = score
                if this_feature != None:
                  good_features.append(this_feature)
                  best_scores.append(best_score)

                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break
        return best_scores[:-1], good_features[:-1]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        scores, features = self.feature_selection(X,y)
        return X[:,features],scores
    
if __name__ == "__main__"
    X,y = make_classification(n_samples=1000,n_features=100)
    X_transformed,scores = GreedyFeatureSelection()(X,y)
                 
                
                    
                    
