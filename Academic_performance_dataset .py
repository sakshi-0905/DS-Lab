# 2) Data Wrangling II 
# Create an “Academic performance” dataset of students and perform the following operations using 
# Python. 
 
# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or 
# inconsistencies, use any of the suitable techniques to deal with them. 
# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques 
# to deal with them. 
# 3. Apply data transformations on at least one of the variables. The purpose of this 
# transformation should be one of the following reasons: to change the scale for better 
# understanding of the variable, to convert a non-linear relation into a linear one, or to decrease 
# the skewness and convert the distribution into a normal distribution. 
 
# Reason and document your approach properly. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Student_ID': [1,2,3,4,5,6,7,8,9,10],
    'Age': [18,19,18,np.nan,20,19,18,21,100,19],
    'Gender': ['Male','Female','Female','Male','Male',None,'Female','Male','Female','female'],
    'Study_Hours': [2,5,np.nan,4,10,3,4,60,5,4],
    'Attendance (%)': [85,90,88,np.nan,95,80,82,100,87,89],
    'Marks': [65,78,70,72,90,60,np.nan,95,68,75]
}

df = pd.DataFrame(data)

print("Missing Values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].mean())
df['Attendance (%)'] = df['Attendance (%)'].fillna(df['Attendance (%)'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Gender'] = df['Gender'].str.capitalize()

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] > upper, upper,
                       np.where(df[col] < lower, lower, df[col]))

df['Study_Hours_Log'] = np.log1p(df['Study_Hours'])

sns.histplot(df['Study_Hours'], kde=True)
plt.title("Study Hours Before Transformation")
plt.show()

sns.histplot(df['Study_Hours_Log'], kde=True)
plt.title("Study Hours After Log Transformation")
plt.show()

print("\nFinal Cleaned Dataset:\n", df)
