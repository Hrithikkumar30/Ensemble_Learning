import pandas as pd
import numpy as  np
import seaborn as sns
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
print(titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean())

titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull() , 0,1)
print(titanic.head())

gender_num = {'male' : 0 , "female" : 1}
titanic['Sex'] = titanic['Sex'].map(gender_num)

titanic.drop(['Cabin' , 'Embarked' , 'Name' , 'Ticket'] , axis=1 , inplace=True)
print(titanic.head())