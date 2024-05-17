###**********************************************************************
### General Flow for Tuning
### Include Training & Validation Split
### select model based on problem Type
### brute force method: very high run time
###**********************************************************************
best_accuracy = 0
best_param = {'a':0,'b':0,'c':0}
for a in range(1,11):
    for b in range(1,11):
        for c in range(1,11):
            #inititalize model with current parameters
            model = MODEL(a,b,c)
            #fir the model
            model.fit(X,y)
            #make predections
            pred = model.predict(validation_data)
            #calculate accuracy
            Accu = model.accuracy_score(targets,preds)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param['a'] = a
                best_param['b'] = b
                best_param['c'] = c

