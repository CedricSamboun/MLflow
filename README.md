# README

# Introduction

This project is focused on building a machine learning model to predict loan return ability using the application_train.csv and application_test.csv datasets. The model is then trained and packaged using MLFLOW. You will find in the repository the notebook MLFLOW2.ipynb containing all the code for this project, that you can also find under multiple python scripts such as models_training.py or data_preparation.py in order to make them more easily shared or tested. 

# Data Preparation

In this step, we prepared the data for training a machine learning model by importing the application_train.csv and application_test.csv datasets, cleaning and transforming the data, handling missing values, and one-hot encoding the categorical variables. This ensured that the data was in a format that the model could effectively use to make accurate predictions.


# Model Building

Here we built a machine learning model using the RandomForestClassifier from the sklearn.ensemble library. We split the data into training and test sets, trained the classifier on the training data, and evaluated its performance on the test data using metrics such as accuracy and ROC AUC score.


# Model Evaluation

For Model Evaluation, we evaluated the performance of the built model. The accuracy of the classifier is 93.93% which is a good one, indicating that the model is performing well in terms of correctly classifying the data.
The ROC AUC score is another metric that is used to evaluate the performance of the classifie is 56.12%. A ROC AUC score close to 1 indicates a good performance, and a score close to 0.5 indicates a not so good performance, indicating that the model is not performing well in terms of correctly classifying the data and it might need some improvement.



# Data visualization

The confusion matrix is visualized using the seaborn.heatmap() method, which is annotated with the annot=True parameter and formatted with the fmt='d' parameter.
The heatmap is saved as an image named 'confusion_matrix.png' using the plt.savefig() method.
Deployment
The trained model is ready to be deployed in a production environment using MLFLOW.
The model can be deployed on various platforms using the mlflow.pyfunc.deploy() method.
The deployed model will be accessible using a REST API.

# Deployment

In this part, we deployed the trained model for production use thanks to mlflow. We used the joblib library to save the model as a pickle file, which can be easily loaded and used for making predictions on new data. Additionally, we used mlflow to log the model, its parameters, accuracy, ROC AUC score, and the confusion matrix, allowing to package the model and monitor its performance. This information can be used to monitor the performance of the model over time and make updates as necessary.



# SHAP


The SHAP library allows us to explain our model by seeing the contribution of the different predictors.

The steps to use this library is to first build a random forest regression model, then we obtain our shap values with the TreeExplainer. We then use our shap values to obtain different summary graphs on all our values, a specific point and for each class that was identified in our dataset.

The first summary graph we obtained is a variable importance plot, it gives us a global interpretability. In this plot the variables at the top are the one with the most impact on the model, meaning they have the most predictive power. So, in our dataset the variables "EXT_SOURCE_3", "EXT_SOURCE_1" and "EXT_SOURCE_2" have the most impact on the model, while "AMT_CREDIT" has the least.

The second graph shows the relations, they may be positives or negatives, between the target variable and the predictors. They are also ordered by higher to lower importance on the model, and show if the variable is associated with a high or low prediction.
