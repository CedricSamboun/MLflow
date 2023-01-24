import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#import preprocessed csv
print("Loading preprocessed train and test csv files")
df_train = pd.read_csv('preprocessed_train.csv')
df_test = pd.read_csv('preprocessed_test.csv')

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('TARGET', axis=1), df_train['TARGET'], test_size=0.2)

#Create the random forest classifier
clf = RandomForestClassifier()

#Train the classifier on the training data
clf.fit(X_train, y_train)

#Evaluate the classifier on the test data
accuracy = clf.score(X_test, y_test)

#accuracy
print('Accuracy: {:.2f}%'.format(accuracy * 100))

#roc_auc_score
print('ROC AUC Score: {:.2f}%'.format(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) * 100))

#confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, clf.predict(X_test))
print("Confusion matrix: \n",conf_mat)

#dataviz
import matplotlib.pyplot as plt
import seaborn as sns

#Create a heatmap
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')

#create confusion matrix.png
plt.savefig('confusion_matrix.png')
print("Confusion matrix image saved as confusion_matrix.png")



