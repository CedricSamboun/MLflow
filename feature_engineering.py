import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# import dataset application_train.csv an application_test.csv
df_train = pd.read_csv('application_train.csv')
df_test = pd.read_csv('application_test.csv')


# Drop rows with NaN values 
print("Data before dropping NaN values:")
print(df_train.head())
df_train = df_train.dropna()
print("Data after dropping NaN values:")
print(df_train.head())

# one-hot encoding of categorical variables
print("Data before one-hot encoding:")
print(df_train.head())
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
print("Data after one-hot encoding:")
print(df_train.head())

# save preprocessed data
df_train.to_csv('preprocessed_train.csv', index=False)
df_test.to_csv('preprocessed_test.csv', index=False)
