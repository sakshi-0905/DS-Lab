'''1) Data Wrangling, I 
Perform the following operations using Python on any open source dataset (e.g., data.csv) 
1. Import all the required Python Libraries. 
2. Locate open source data from the web (e.g., https://www.kaggle.com). Provide a clear 
description of the data and its source (i.e., URL of the web site). 
3. Load the Dataset into pandas dataframe. 
4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe() 
function to get some initial statistics. Provide variable descriptions. Types of variables etc. 
Check the dimensions of the data frame. 
5. Data Formatting and Data Normalization: Summarize the types of variables by checking the 
data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set. 
If variables are not in the correct data type, apply proper type conversions. 
6. Turn categorical variables into quantitative variables in Python. 
 
In addition to the codes and outputs, explain every operation that you do in the above steps and explain 
everything that you do to import/read/scrape the data set. '''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None, names=columns)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Shape (Rows, Columns):")
print(df.shape)

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nDataset Info:")
print(df.info())

df['species'] = df['species'].astype('category')

df.dropna(inplace=True)

print("\nShape After Cleaning Missing Values:")
print(df.shape)

scaler = MinMaxScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

encoder = LabelEncoder()
df['species_encoded'] = encoder.fit_transform(df['species'])

print("\nFinal Dataset Preview:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)
