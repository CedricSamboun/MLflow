import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# import dataset application_train.csv an application_test.csv
df_train = pd.read_csv('application_train.csv')
df_test = pd.read_csv('application_test.csv')


# Description of training data
print("Description of training data:")
print(df_train.describe())

# Description of testing data
print("Description of testing data:")
print(df_test.describe())

# Shape of training data
print("Shape of training data:")
print(df_train.shape)

# Shape of testing data
print("Shape of testing data:")
print(df_test.shape)

# Distribution of target labels in training data
print("Distribution of target labels in training data:")
sns.countplot(df_train.TARGET)
print(df_train['TARGET'].value_counts())
plt.show()
