import pandas as pd
import numpy as  np
import seaborn as sns
import joblib
from matplotlib import pyplot as plt
titanic = pd.read_csv("Titanic-Dataset.csv")
# print(titanic.head())

#cleaning the data (feature engeneering)
# print(titanic.isnull().sum())
titanic['Age'].fillna(titanic['Age'].mean() , inplace=True)
# print(titanic.head(10))

for i , col in enumerate (['SibSp' , 'Parch']):
    plt.figure(i)
    sns.catplot(x=col , y = 'Survived' , data=titanic , kind='point' , aspect=2,)


titanic['Family_count'] = titanic['SibSp'] + titanic['Parch']
titanic.drop(['PassengerId' , 'SibSp' , 'Parch'] , axis=1 , inplace=True)


# <---------------------- Cleaning catogerical Data ---------------------->
# print(titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean())

titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull() , 0,1)
# print(titanic.head())

gender_num = {'male' : 0 , "female" : 1}
titanic['Sex'] = titanic['Sex'].map(gender_num)

titanic.drop(['Cabin' , 'Embarked' , 'Name' , 'Ticket'] , axis=1 , inplace=True)
# print(titanic.head())

# <------------------ Train-Test Data Spilitting -------------------->

from sklearn.model_selection import train_test_split

features = titanic.drop('Survived' , axis=1)
labels= titanic['Survived']

X_train , X_test , Y_train , Y_test = train_test_split(features,labels,test_size=0.4 , random_state=42)
X_val , X_test , Y_val , Y_test = train_test_split(X_test,Y_test,test_size=0.5 , random_state=42)


# for dataset in (Y_train, Y_val,Y_test):
#     # print(round(len(dataset) / len(labels) , 2))
    
    
    
    
from sklearn.model_selection import GridSearchCV  #GridSearchCV will help us fit and evaluate a model from scikit-learn.
from sklearn.ensemble import GradientBoostingClassifier , AdaBoostClassifier

def print_result(results):
    print('Best parameters: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']

    for mean , std , params in zip (means, stds, results.cv_results_['params']):
        print ('{}(+/- {}) for {}'.format(round(mean , 3) ,round(std * 2 ,3),params))
        
        
gb = GradientBoostingClassifier()
Parameters = {
    'n_estimators': [5,50,250,500],
    'max_depth' : [1,3,5,7,9],
    'learning_rate' : [0.01,0.1,10,100]
}

cv = GridSearchCV(gb, Parameters , cv=5)
cv.fit(features , labels.values.ravel())

print_result(cv)

print(cv.best_estimator_)

joblib.dump(cv.best_estimator_ , '../../../models/GB_model.pkl')