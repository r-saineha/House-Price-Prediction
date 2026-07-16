## House Price Prediction Using Machine Learning
## Overview

This project is a Machine Learning and Data Analytics application that predicts house prices based on various property features. The model is built using a Random Forest Regressor with a Scikit-learn Pipeline for preprocessing and prediction. A Streamlit web application provides an interactive interface where users can enter house details, receive an estimated price, and explore the dataset through visual analytics.

## Features
- Predict house prices using Machine Learning
- Interactive Streamlit web application
- Data preprocessing using Scikit-learn Pipeline
- Automatic handling of numerical and categorical features
- Missing value imputation
- One-Hot Encoding for categorical variables
- Random Forest Regression model
- Feature importance visualization
- Exploratory Data Analysis (EDA)
- Interactive charts and graphs
- Correlation heatmap
- Dataset summary statistics
- Download prediction results
## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
## Dataset Features

## The model predicts house prices using features such as:

- Bedrooms
- Bathrooms
- Living Area
- Lot Area
- Floors
- Waterfront
- View Rating
- House Condition
- Above Ground Area
- Basement Area
- Year Built
- Year Renovated
- Street
- City
- State Zip
- Country
- Sale Date (converted into Year, Month, and Day)
## Machine Learning Workflow
- Load and preprocess the dataset.
- Handle missing values using SimpleImputer.
- Encode categorical features using OneHotEncoder.
- Split the dataset into training and testing sets.
- Train a Random Forest Regressor.
- Evaluate the model using MAE, RMSE, and R² Score.
- Save the trained model using Joblib.
- Deploy the model with Streamlit for real-time predictions.
## Data Analytics

The application includes multiple visualizations to help understand the dataset:

- House Price Distribution
- Correlation Heatmap
- Bedrooms vs Price
- Bathrooms vs Price
- Living Area vs Price
- House Condition Distribution
- Waterfront Analysis
- Top 10 Cities by Number of Houses
- Feature Importance Chart
- Project Structure
House-Price-Prediction/
│
├── app.py
├── train.py
├── data (1).csv
├── house_price_model.pkl
├── requirements.txt
└── README.md
## Model Evaluation

The trained model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Future Improvements
- Interactive Plotly dashboards
- House location visualization using maps
- Hyperparameter tuning
- Model comparison (XGBoost, Gradient Boosting, Linear Regression)
- Deployment on Streamlit Community Cloud
- Prediction history and user authentication
