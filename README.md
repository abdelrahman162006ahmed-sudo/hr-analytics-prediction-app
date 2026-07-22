HR Intelligence & Analytics System


🤖 HR Analytics Prediction App

An end-to-end Machine Learning application built with Python, Scikit-learn, XGBoost, and Streamlit to support Human Resources teams in making smarter hiring and employee retention decisions.

The system provides two intelligent prediction modules:

Employee Attrition Prediction – Predicts whether an employee is likely to leave the company based on demographic and professional attributes.
Salary Prediction – Estimates the expected salary of a new candidate using information extracted from their CV, such as education, experience, skills, certifications, company size, industry, work location, and remote work status.
🚀 Project Overview

Employee turnover is one of the most expensive challenges for organizations. This project helps HR departments identify employees who are at risk of leaving before resignation occurs, enabling proactive retention strategies.

In addition, the application predicts an appropriate salary for new candidates, helping recruiters maintain fair and consistent compensation during the hiring process.

The application combines machine learning models with an interactive Streamlit interface to deliver real-time predictions in a user-friendly environment.

✨ Key Features
Employee Attrition Prediction
Salary Prediction from Candidate Information
Interactive Streamlit Dashboard
Real-time Confidence Scores
Input Validation to prevent unrealistic values
Trained Machine Learning Models
Saved Models & Encoders using Joblib
Clean and Responsive User Interface
🧠 Machine Learning Pipeline
Employee Attrition Prediction

The employee attrition dataset contained a noticeable class imbalance, where employees who left the company represented the minority class.

To improve the model's ability to recognize both classes fairly, the imbalance was addressed using:

SMOTE (Synthetic Minority Oversampling Technique)

This generated synthetic samples for the minority class, resulting in a more balanced training dataset and improved classification performance.

The classification model uses an ensemble approach consisting of:

XGBoost Classifier
Random Forest Classifier
Soft Voting Classifier

Additional preprocessing included:

Label Encoding for categorical features
Standard Scaling
Train/Test Split with Stratification
Salary Prediction

The salary estimation model was trained on a dataset containing more than 250,000 records, allowing the model to learn salary patterns across different industries, education levels, company sizes, locations, and experience levels.

The regression model is based on:

XGBoost Regressor

Feature preprocessing included:

Label Encoding
Text normalization
Feature engineering
Model serialization using Joblib
🛠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Joblib
Streamlit
📊 Workflow
Data preprocessing
Handling missing values and categorical encoding
Balancing the employee dataset using SMOTE
Feature scaling
Model training
Model evaluation
Model serialization
Deployment with Streamlit
🎯 Business Value

This application assists HR professionals by:

Predicting employee resignation risk before it happens.
Supporting employee retention strategies.
Providing data-driven salary recommendations for new candidates.
Improving recruitment consistency and decision-making.
Reducing hiring costs and employee turnover.
📌 Future Improvements
CV parsing using NLP.
Explainable AI with SHAP values.
Cloud deployment.
HR analytics dashboard with interactive visualizations.
Automated retraining pipeline.
Database integration for employee records.
👨‍💻 Author

Abdelrahman Ahmed

Machine Learning Engineer | Python Developer
