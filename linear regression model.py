import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
# Load Boston Housing dataset from CSV 
df = pd.read_csv('BostonHousing.csv') 
print('=== Dataset Info ===') 
print(df.head()) 
print('\nShape:', df.shape) 
print('\n=== Statistical Summary ===') 
print(df.describe()) 
print('\n=== Missing Values ===') 
print(df.isnull().sum()) 

# Features and Target (medv = Median House Value) 
X = df.drop('medv', axis=1) 
y = df['medv'] 
# Train-Test Split (80-20) 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42
    ) 
print('\nTraining set size:', X_train.shape) 
print('Testing set size :', X_test.shape) 
# Train Linear Regression Model 
model = LinearRegression() 
model.fit(X_train, y_train) 
# Predict 
y_pred = model.predict(X_test) 
# Model Evaluation 
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred) 
print('\n=== Model Evaluation ===') 
print(f'MAE : {mae:.4f}') 
print(f'MSE : {mse:.4f}') 
print(f'RMSE : {rmse:.4f}') 
print(f'R^2 : {r2:.4f}') 
# Coefficients 
print('\n=== Feature Coefficients ===') 
coef_df = pd.DataFrame({
    'Feature': X.columns, 
    'Coefficient': model.coef_
    }) 
print(coef_df.to_string(index=False)) 
print(f'\nIntercept: {model.intercept_:.4f}') 
# Sample Predictions vs Actual 
print('\n=== Sample Predictions vs Actual ===') 
results = pd.DataFrame({
    'Actual': y_test[:10].values, 
    'Predicted': y_pred[:10].round(2)
    }) 
print(results.to_string(index=False))