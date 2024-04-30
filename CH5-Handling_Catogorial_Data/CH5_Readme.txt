Chapter 5 talks about How to handle categorial Data & also about below items

1) How to organize an ML Project - this helps to build & test quick & dirty models fast
	This will be improved in future check-ins with more customizations
2) This chapter also talks about how to handle categorial data & test various models with 
   different encoding types. Although it doesn't cover target encoding & Entity embedding which 
will be part of future check-ins
	Data : We have used 2 different datasets
		1) Cat in the dat II
		2) Adult dataset from US census 
	Hyper-tuning : N/A - These approaches are meant to be simple tests without much 
			     hyper-tuning
3) Tested multiple approaches
	All the below approaches uses stratified Fold & AUC as evaluation criteria 
	Train - Cat in the dat II 
	Train2 - Adult dataset
	a) One hot encoding with Logistic regression
	b) Label encoding with Random Forest
	c) Label encoding with XGB
	d) One hot encoding with Sparse; SVD decomposition with Random Forest
	e) Label encoding with XGB - Only Categorial features
	f) Label encoding with XGB - Both Categorial & Numerical features
	g) Feature engineering with simple 2 combination feature generation with XGB; Label 
	   encoding  
 

 

