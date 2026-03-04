# ML-aIrIEEEna-Solution
Detecting Device Faults with Machine Learning

Project Overview:
This project makes a fault detection system for devices that uses sensor data and machine learning.
The goal is to figure out if a device is working normally (Class = 0) or not (Class = 1).

There are 47 numerical features (F01–F47) in the dataset that show operational parameters that were collected by a monitoring system. These features record the internal states of the device and how it interacts with its surroundings during activity cycles.

We used a supervised machine learning method to train models that can use these features to find device problems.

Problem Statement:
Given a set of measurements from device sensors
Input: 47 numbers (F01–F47)
Output: A binary classification
Class 0 means the device is working normally and 1 means the device has a problem.

The goal is to create a model that accurately

Dataset Description: 
Training Dataset (TRAIN.csv) has:

ID: a number that is unique to each record
F01–F47: values for sensor features
Class = target variable

Test Dataset (TEST.csv) has:

ID: a unique identifier
F01–F47 are the values of the features.
The Class column is not in the test dataset, so you have to use the trained model to make predictions.

Machine Learning Models Utilized 

Random Forest Classifier 
A tree-based ensemble method that builds multiple decision trees and aggregates predictions.

XGBoost Classifier
An optimized gradient boosting algorithm that improves performance by sequentially correcting errors from previous trees.

Both models were evaluated using:

Accuracy
Precision
Recall
F1-score

XGBoost showed slightly better performance for this dataset.

Project Structure:
ML-aIrIEEEna-Solution/
│
├── alrIEEEna_26_dataset/
│   ├── TRAIN.csv
│   ├── TEST.csv
│   ├── train_xgboost.py
│
├── FINAL.csv
├── README.md

Installation and Setup
1. Clone the Repository "https://github.com/ShubhamJoshi01/ML-aIrIEEEna-Solution"
2. Install Dependencies
Install the required Python libraries:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost


Usage Instructions
Run the training and prediction script:

python 1xgboost.py
The script will:
Load the training dataset
Train the machine learning model
Evaluate model performance
Generate predictions for the test dataset
Create the submission file

Output File
The script generates a file named: FINAL.csv

Format:
ID,CLASS
1,1
2,0
3,0
4,1

Requirements:
Must contain ID and CLASS columns
Must maintain the same order as TEST.csv
Number of rows must match the test dataset
