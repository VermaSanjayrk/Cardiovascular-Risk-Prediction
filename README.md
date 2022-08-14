# Cardiovascular-Risk-Prediction
## INTRODUCTION
Coronary artery disease (CAD) is the most common heart disease seen in today’s population worldwide. Recent studies of the American Heart Association have shown that coronary artery diseases recorded 13% death in the USA in 2018 and worldwide in 2015. Amongst other diseases CAD found to be one of the most common causes of death, with the record of 15.6% of all results across the globe. Because this disease is associated with modifiable risk factors which indirectly related with lifestyle and intervention, timing of detection and diagnostic accuracy are especially relevant in clinical management of patients with CAD.
Over the past years, approaches making a significant impact in the detection and diagnosis of diseases that include machine learning (ML). In general, ‘training’ an algorithm with a control dataset for which the disease status (disease or no disease) is known, and then applying this trained algorithm to a variable dataset in order to predict the disease status in patients for whom it is not yet determined. The ML algorithm will be better trained as a predictor for disease status as larger cohorts of data are introduced. prediction with ML would empower clinicians with improved detection, diagnosis, classification, risk stratification and ultimately, management of patients, all while potentially minimizing required clinical intervention are More accurate.
## PROBLEM STATEMENT
The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD).
## DATA DESCRIPTION
### Attribute Description :
Each attribute is a potential risk factor. There are both demographic, behavioral, and medical risk factors.

### Demographic:
• Sex: male or female("M" or "F")
• Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
### Behavioral :
• is_smoking: whether or not the patient is a current smoker ("YES" or "NO")
• Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)

### Medical( history)
• BP Meds: whether or not the patient was on blood pressure medication (Nominal)
• Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
• Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
• Diabetes: whether or not the patient had diabetes (Nominal)
### Medical(current)
• Tot Chol: total cholesterol level (Continuous)
• Sys BP: systolic blood pressure (Continuous)
• Dia BP: diastolic blood pressure (Continuous)
• BMI: Body Mass Index (Continuous)
• Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of a large number of possible values.)
• Glucose: glucose level (Continuous)
•Predict variable (desired target)
10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”) - Dv
## DATASET PREPROCESSING 
According to the size of the dataset, extensive cleaning of the dataset is not needed. Column names in the dataset are short as we get a meaningful  explanation of  these features. Dataset contains NaN values and it is removed by replacing the NAN values by unknown values.
## APPROACH
We checked the Outliers and correlation matrix to overcome the noise in the dataset. Also, data was balanced using the SMOTE method and scaled by Standard Scaler transformation. As the coronary heart diseases dataset defines the classification problem. We decided to train the models such as Logistic regression, K-nearest Neighbors, Decision Tree Classifier & Support Vector Machine. Also, we used Hyperparameter Tuning for improvement in the model fitting to understand the better results of the model as well as the metrics.
## DATA INSIGHTS
### Describe
![image](https://user-images.githubusercontent.com/91052155/184524084-d2a81100-b05e-4b38-b95f-bb6a0daee08d.png)
### Visualize the presence of NAN values-
![image](https://user-images.githubusercontent.com/91052155/184524094-78cd3d2b-4c33-47d3-8cc9-6a0f2246f205.png)
NAN values are removed by replacing it with unknown values with missing no matrix. The above figure shows that presence of NAN values will not reflect in the dataset.
## OUTLIERS DETECTION
An outlier is a data point that is noticeably different from the rest. They represent errors in measurement, bad data collection, or simply show variables not considered when collecting the data.
![image](https://user-images.githubusercontent.com/91052155/184524111-8afc3ddf-8cf4-43c9-8d8f-8c7e0600e1a1.png)
## CORRELATION MATRIX (HEATMAP)
![image](https://user-images.githubusercontent.com/91052155/184524122-c5f18275-427e-4972-8a51-cec46e200b27.png)
Heart Rate and prevalent Stroke are the lowest correlated with the target variable.
Also, some of the features have a negative correlation with the target value and some have positive.
## DATA MODELING
After the data preparation is completed, it is ready for the purpose of analysis. Only numerical valued features are taken into consideration. The data were combined and labeled as X and y as independent and dependent variables respectively. 
## SPLITTING THE DATASET
The train_test_split was imported from the sklearn.model_selection. The data is now divided into 70% and 30% as train and test splits respectively. 70% of the data is taken for training the model and 30% is for a test and the random state was taken as 0. 
## MODEL IMPLEMENTATION
### ➔	Support Vector Machine 
Support Vector Machine (SVM) is a classification technique used for the classification of linear as well as non-linear data. SVM is the margin based classifier. It selects the maximum margin. This model is further used to perform classification of testing data.
![image](https://user-images.githubusercontent.com/91052155/184524174-40061f85-7e22-428b-b74c-da9c2322e2c0.png)
### ➔	Decision Tree Classifier 
Decision tree build classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.
![image](https://user-images.githubusercontent.com/91052155/184524188-c84a9c0f-2e0c-4ff7-bab9-dca44d616f55.png)
## Hyperparameter Tuning on K-nearest Neighbors algorithm-
![image](https://user-images.githubusercontent.com/91052155/184524205-8c36d70e-d9c7-43cb-a5f6-8257d693d649.png)
![image](https://user-images.githubusercontent.com/91052155/184524211-0c3e3726-1a99-4a94-ae15-79e052929346.png)
## ROC Curve: 
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:
a)	True Positive Rate-True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:TPR=TP / (TP+FN)
b)	False Positive Rate-False Positive Rate (FPR) is defined as follows:FPR =FP / (FP+TN)
![image](https://user-images.githubusercontent.com/91052155/184524235-9537fcc6-2a7f-49ff-a341-b8711cc52ab6.png)
## METRICS USED 
●	Classification Report:A classification report is a performance evaluation metric in machine learning. It is used to show the precision, recall, F1 Score, and support of your trained classification model. 
1) Accuracy: the proportion of total dataset instances that were correctly predicted out of the total instances
accuracy=(true positives+true negatives)/total
2) Recall (sensitivity): the proportion of the predicted positive dataset instances out of the actual positive instances
sensitivity=true positives/(true positives+false negatives)

3) F1 score: a composite harmonic mean (average of reciprocals) that combines both precision and recall. For this, we first measure the precision, the ability of the model to identify only the relevant dataset instances
precision=true positives/(true positives+false positives)
The F1 score is estimated as
F1=2×(precision×recall)/(precision+recall)
●	Confusion Matrix:A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted by the machine learning model. This gives us a holistic view of how well our classification model is performing and what kinds of errors it is making.
 
●	True Positive (TP) -The predicted value matches the actual value.The actual value was positive and the model predicted a positive value
●	True Negative (TN)  -The predicted value matches the actual value.
The actual value was negative and the model predicted a negative value
●	False Positive (FP) – Type 1 error :-The predicted value was falsely predicted.The actual value was negative but the model predicted a positive value
●	False Negative (FN) – Type 2 error:-The predicted value was falsely predicted.The actual value was positive but the model predicted a negative value
## CONCLUSION
Patients of age group 32 to 70 years are present among which 38 to 46 years age group have high smoking habits.
➢	 Comparative to male patients’ female patients is more.
➢	 Across the dataset we have 1307 male patients, of which 809 male patients smoke cigarettes.
➢	 We have 1620 female patients out of which 638 smoke cigarettes.
➢	 patients who previously had a stroke and Number of patients with medical history like blood pressure medication, Diabetes is very low.
➢	 Logistic Regression-Nearest Neighbors, Support Vector Machine & Decision Tree Classifier models were implemented.

➢	 From above these models, we found that KNN is the best fitted model compared to other models

➢	 In Hyperparameter tuning, we observed that K-Nearest Neighbors accuracy has improved which shows that KNN (with Hyperparameter Tuning) is the best fitted model for coronary heart disease dataset.
Train Accuracy = 85.30 & Test Accuracy = 84.07

For future improvement in the model fitting for coronary heart disease, we can perform the Random Forest Classifier, XGBoost models also. Consulting medical people we can analyze the feature in proper and required manner to approach the disease cause and effects.

# Thank You


