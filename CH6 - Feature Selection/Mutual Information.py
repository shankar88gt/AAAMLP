"""
The metric we'll use is called "mutual information". Mutual information is a lot like correlation 
in that it measures a relationship between two quantities. The advantage of mutual information is 
that it can detect any kind of relationship, while correlation only detects linear relationships.

Mutual information is a great general-purpose metric and especially useful at the start of feature development 
when you might not know what model you'd like to use yet. It is:
1) easy to use and interpret,
2) computationally efficient,
3) theoretically well-founded,
4) resistant to overfitting, and,
5) able to detect any kind of relationship

Mutual information describes relationships in terms of uncertainty. The mutual information (MI) between two 
quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. 
If you knew the value of a feature, how much more confident would you be about the target?

Technical note: What we're calling uncertainty is measured using a quantity from information theory known as 
"entropy". The entropy of a variable means roughly: "how many yes-or-no questions you would need to describe 
an occurance of that variable, on average." The more questions you have to ask, the more uncertain you must 
be about the variable. Mutual information is how many questions you expect the feature to answer about the target


Mutual information increases as the dependence between feature and target becomes tighter. 
Mutual information can capture any kind of association (not just linear, like correlation.)

Here are some things to remember when applying mutual information:
1) MI can help you to understand the relative potential of a feature as a predictor of the target, 
considered by itself. 
2) It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. 
MI can't detect interactions between features. It is a univariate metric.
3) The actual usefulness of a feature depends on the model you use it with. 
A feature is only useful to the extent that its relationship with the target is one your model can learn. 
Just because a feature has a high MI score doesn't mean your model will be able to do anything with that 
information. You may need to transform the feature first to expose the association.


The scikit-learn algorithm for MI treats discrete features differently from continuous features. 
Consequently, you need to tell it which are which. As a rule of thumb, anything that must have a float dtype is
not discrete. Categoricals (object or categorial dtype) can be treated as discrete by giving them a 
label encoding. 


https://www.kaggle.com/code/ryanholbrook/mutual-information

"""