import pandas as pd
import numpy as  np
import seaborn as sns
from matplotlib import pyplot as plt
titanic = pd.read_csv("Titanic-Dataset.csv")
print(titanic.head())

#cleaning the data (feature engeneering)
print(titanic.isnull().sum())