####**********************************************************************************************
#### Unnivariate Feature Selection
####**********************************************************************************************
###  scoring each feature against target variable
### Mutual Information, ANOVA F test & Chi**2 
### Chi**2 only for non negative data 

### Select K best - top K scoring features
### keep top features based on % specified by user

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelection:
    def _init_(self,n_features,problem_type,scoring):
        ### n_features : selectpercentile if float else SelectKbest
        ### Problem Type : classification or Regression
        ### scoring : scoring function
        if problem_type == "classification":
            valid_scoring = {
                "f_classif" : f_classif,
                "chi2" : chi2,
                "mutual_info_classif" : mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression" : f_regression,
                "mutual_info_regression" : mutual_info_regression                
            }
        if scoring not in valid_scoring:
            raise Exception("Invalid Scoring function")
        
        if isinstance(n_features, int):
            self.selection = selectKbest(valid_scoring[scoring],k=n_features)
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(valid_scoring[scoring],percentile=(n_features*100))
        else: raise Exception("Invalid Type of feature")

        def fit(self,X,y):
            return self.selection.fit(X,y)
        
        def transform(self,X):
            return self.selection.transform(X)

        def fit_transform(self,X,y):
            return self.selection.fit_transform(X,y)


### Sample example 
ufs = UnivariateFeatureSelection(n_features=0.1,problem_type='regression',scoring='f_regression')
ufs.fit(X,y)
ufs.transformed = ufs.transform(X)
