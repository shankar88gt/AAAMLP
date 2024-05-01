###***************************************************************************************************
### Step 1 - remove all features with no variance; these are like constants and doesnt add any value
###
### transformed data will have all columns with variance less
### than 0.1 removed
### **************************************************************************************************
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import VarianceThreshold
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
Y = data["target"]
df = pd.DataFrame(data=X,columns=col_names)
print(df.head())
var_tresh = VarianceThreshold(threshold=0.1)
transformed_data = var_tresh.fit_transform(df)
print(var_tresh.get_support(indices=True))
df1 = pd.DataFrame(data=transformed_data)
print(df1.head())
#print(transformed_data.columns.to_list())


