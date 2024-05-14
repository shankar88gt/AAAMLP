###***************************************************************************************************
### Step 1 - remove all features with no variance; these are like constants and doesnt add any value
### transformed data will have all columns with variance less than threshold removed
### **************************************************************************************************
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
print(col_names)
Y = data["target"]
df = pd.DataFrame(data=X,columns=col_names)
var_tresh = VarianceThreshold(threshold=10)
transformed_data = var_tresh.fit_transform(df)
df1 = pd.DataFrame(data=transformed_data,columns=[list(compress(col_names, var_tresh.get_support(indices=False)))])
print(df1.head())


