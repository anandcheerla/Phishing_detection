
#importing necessary modules
import numpy as np
import pandas as pd
from time import time
from IPython.display import display 
import seaborn as sb
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import matthews_corrcoef

# Pretty display for notebooks
# %matplotlib inline



# Load the PhishingData dataset and store it in data
data = pd.read_csv("PhishingData.csv")

# display
# display(data.head(n=5))


# data.describe()


#Number of instances of different classes of output label.
# print(len(data[data.Result==0]))
# print(len(data[data.Result==-1]))
# print(len(data[data.Result==1]))



#separate the Output label and input features
#store output label as result and input features as features
result=data['Result']
features=data.drop('Result',axis=1)

#plot for the output label classes
sb.countplot(data['Result'])


#Relation between input features and output label classes.
for i in features.columns:
    plt.figure(figsize=(10,6))
    plt.title('%s'%i)
    sb.countplot(data[i],hue=data['Result'])


# print(data.corr())
sb.heatmap(data.corr(),annot=True)



# Split the 'features' and 'result' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    result, 
                                                    test_size = 0.2, 
                                                    random_state = 5)




# Benchmark model
#import logistic regression which is used as a benchmark model
#import Evaluation metrics

#create logistic regression object
clf_lr = LogisticRegression(random_state=5)
#Train the model using training data 
clf_lr=clf_lr.fit(X_train,y_train)

#Test the model using testing data
predictions = clf_lr.predict(X_test)

# print("f1 score is ",f1_score(y_test,predictions,average='weighted'))
# print("matthews correlation coefficient is ",matthews_corrcoef(y_test,predictions))

# #secondary metric,we should not consider accuracy score because the classes are imbalanced.
# print("Accuracy score is ",accuracy_score(y_test,predictions))


def predict(learner, X_train, y_train, X_test, y_test): 
    
    results = {}
    
    learner = learner.fit(X_train,y_train)
    predictions_test = learner.predict(X_test)
    
    results['f1_score'] = f1_score(y_test,predictions_test,average='weighted')
    results['mcc_score'] = matthews_corrcoef(y_test,predictions_test)   
    results['acc_score'] = accuracy_score(y_test,predictions_test)
    
    return results


# TODO: Initialize the three models
clf_A = AdaBoostClassifier(random_state=0)
clf_B = SVC(random_state=0)
clf_C = GradientBoostingClassifier(n_estimators=250,random_state=0)

# print("AdaBoostClassifier : ",predict(clf_A,X_train, y_train, X_test, y_test))
# print("svm : ",predict(clf_B,X_train, y_train, X_test, y_test))
# print("GradientBoostingClassifier : ",predict(clf_C,X_train, y_train, X_test, y_test))



clf = GradientBoostingClassifier(n_estimators=250,random_state=0)

parameters = {'n_estimators':[100,50,250],'learning_rate':[0.1,0.5]}

scorer = make_scorer(f1_score,average='weighted')
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
grid_fit = grid_obj.fit(X_train,y_train)
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
# predictions = (clf_C.fit(X_train, y_train)).predict(X_test)
#best_predictions = best_clf.predict(X_test)

# best_predictions = best_clf.predict(df_inp)

# print("Result is------------")
# print(best_predictions)


# print("Unoptimized model\n------")
# print("f1_score core on testing data: {:.4f}".format(f1_score(y_test, predictions,average='weighted')))
# print("\nOptimized Model\n------")
# print("Final f1_score on the testing data: {:.4f}".format(f1_score(y_test, best_predictions,average='weighted')))


# print(best_clf)