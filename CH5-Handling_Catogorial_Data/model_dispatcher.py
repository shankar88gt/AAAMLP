# model_dispatcher.py
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model

model = {
         "decision_tree_gini": "tree.DecisionTreeClassifier(criterion='gini')",
         "decision_tree_entropy": "tree.DecisionTreeClassifier(criterion='entropy')",
         "LogisticRegression":linear_model.LogisticRegression(),
         "rf":"ensemble.RandomForestClassifier()"
        }