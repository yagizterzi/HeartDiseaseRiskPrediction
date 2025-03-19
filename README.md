Heart Disease Risk Prediction
=============================

This project implements a machine learning model to predict the risk of heart disease based on various health metrics. The system uses clustering to identify risk patterns and a Random Forest classifier to make predictions.

Project Overview
----------------

The Heart Disease Risk Prediction system analyzes various health indicators to predict a person's risk of heart disease, categorizing it into different risk levels (None, Very Low, Low, Medium, High). The project uses the UCI Heart Disease dataset and applies machine learning techniques to make accurate predictions.

Features
--------

*   **Data Preprocessing**: Handles categorical features using One-Hot Encoding and standardizes numerical features
    
*   **Clustering Analysis**: Uses K-means clustering with elbow method to identify natural groupings in the data
    
*   **Model Training**: Employs Random Forest classification with hyperparameter tuning via GridSearchCV
    
*   **Web Interface**: Simple Flask web application for users to input their health metrics and receive risk assessments
    
*   **Visualization**: Includes plots of clusters and model performance metrics
    

Technical Details
-----------------

### Dataset

The project uses the UCI Heart Disease dataset with the following features:

*   Age
    
*   Sex
    
*   Chest Pain Type
    
*   Resting Blood Pressure
    
*   Cholesterol
    
*   Fasting Blood Sugar
    
*   Resting Electrocardiograph Results
    
*   Maximum Heart Rate
    
*   Exercise Induced Angina
    
*   ST Depression Induced by Exercise
    
*   Peak Exercise ST Segment Slope
    
*   Number of Major Vessels
    
*   Thalassemia
    

### Model Pipeline

1.  **Data Preprocessing**:
    
    *   One-Hot Encoding for categorical features
        
    *   Standardization of all features
        
2.  **Clustering**:
    
    *   K-means clustering to identify natural risk groups
        
    *   Elbow method to determine optimal number of clusters
        
3.  **Classification**:
    
    *   Random Forest Classifier trained on cluster assignments
        
    *   Hyperparameter optimization using GridSearchCV
        

Installation and Usage
----------------------

### Prerequisites

*   Python 3.7+
    
*   Required libraries: scikit-learn, pandas, numpy, flask, matplotlib, seaborn, joblib
    

### Setup

1.  Clone the repository:
    

git clone https://github.com/yagizterzi/HeartDiseaseRiskPrediction

1.  Install dependencies:
    

pip install -r requirements.txt

Either use models and run app.py directly or run train.py for creating new model and use that on app.py
    

### Using the Web Interface

1.  Enter your health information in the form
    
2.  Submit the form to get your heart disease risk prediction
    
3.  The system will display your risk level based on the trained model
    

Project Structure
-----------------

*   [train.py]: Script for data preprocessing, model training, and evaluation
    
*   [app.py]: Flask web application for prediction
    
*   [models]: Directory containing saved models
    
    *   [scaler.pkl]: Trained StandardScaler for feature normalization
        
    *   [best\_rf\_model.pkl]: Trained Random Forest model
        
    *   [preprocessor.pkl]: Feature preprocessing pipeline
        

Results
-------

The model achieves good classification performance, with detailed metrics provided in the training output. The clustering approach helps identify natural risk groups in the data, enhancing the interpretability of the predictions.

Future Improvements
-------------------

*   Adding more sophisticated models and ensemble techniques
    
*   Incorporating feature importance analysis
    
*   Enhancing the web interface with visualization of risk factors
    
*   Adding explanations for predictions
    
*   Deployment to a cloud platform for wider accessibility
    

License
-------

This project is licensed under the MIT License - see the [LICENSE] file for details.

Acknowledgments
---------------

*   UCI Machine Learning Repository for the heart disease dataset
    
*   scikit-learn developers for the machine learning tools
    
*   Flask framework for the web application
