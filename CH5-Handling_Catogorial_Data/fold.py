#fold.py
import pandas as pd
from sklearn import model_selection
import config

# read the training file 

print(config.TRAINING_FILE)
df=pd.read_csv(config.TRAINING_FILE)

# Set Kfold value for each entry to be -1
df['kfold'] = -1

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

print(df.shape)

# extract target values
y = df.target.values

# stratified Kfolds in case target variable distribution is sckewed
kf= model_selection.StratifiedKFold(n_splits=config.FOLD)
for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold'] = f

#Stats
print(df.shape)
print(df['kfold'].value_counts())
print(df.groupby(['kfold','target'])['target'].count())

# Save the Training file 
df.to_csv(config.TRAINING_FILE_FOLDS,index=False)

print('Done')


