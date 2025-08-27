# S-P500_ML-DL-Model-with Aj Ann 

# Stock Price Prediction & Direction Analysis

## Overview
This project demonstrates stock price prediction using Machine Learning models (Random Forest, XGBoost, LightGBM, KNN, Decision Tree,LSTM, Transformer) and direction classification. The workflow includes:

1. **Data Preparation**
   - Load historical stock data (Date, Close, High, Low, Volume, Symbol)
   - Split into train/test set (80/20)
   - Handle missing values with linear interpolation
   - Feature engineering: SMA, EMA, MACD, Trend streak
   - Standardization of indicators

2. **Feature Engineering**
   - Technical indicators:
     - SMA (Simple Moving Average)
     - EMA (Exponential Moving Average)
     - MACD & MACD Signal
     - Trend & Trend Streak
   - Lag features can be added for improved prediction
   - Scaling to prevent bias and reduce outlier effects

3. **Model Training**
   - Regression models:
     - Random Forest Regressor
     - XGBoost Regressor
     - LightGBM Regressor
   - Target: `Close` price of the next day
   - Evaluation metrics: MAE, RMSE, R²

4. **Direction Classification**
   - Convert regression predictions to binary direction:
     - 1 = Price up
     - 0 = Price down
   - Evaluation metrics:
     - F1-score
     - Intersection over Union (IoU)
     - Confusion Matrix

5. **Visualization**
   - Plot actual vs predicted stock prices
   - Highlight differences in direction prediction
   - Optional: zoom in on recent days to see prediction accuracy

## Installation


เติมค่าด้วยวิธี Linear Interpotation
<img width="968" height="661" alt="image" src="https://github.com/user-attachments/assets/eeb702f8-9b09-4044-b388-52cb497efc5a" />

<img width="1160" height="547" alt="image" src="https://github.com/user-attachments/assets/7c7f916a-33fb-4672-8fbc-a771d426dcbd" />

<img width="552" height="413" alt="image" src="https://github.com/user-attachments/assets/b0a790ae-c219-4722-88eb-87fdaa3a13c5" />



```bash
# Install required packages
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm
