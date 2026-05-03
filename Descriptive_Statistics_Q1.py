# 3) Descriptive Statistics - Measures of Central Tendency and variability 
# Perform the following operations on any open source dataset (e.g., data.csv) 
# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a dataset (age, income etc.) with numeric variables grouped by one of the qualitative 
# (categorical) variable. For example, if your categorical variable is age groups and quantitative 
# variable is income, then provide summary statistics of income grouped by the age groups. 
# Create a list that contains a numeric value for each response to the categorical variable. 

import pandas as pd

data = {
    'Age_Group': ['Young', 'Young', 'Adult', 'Adult', 'Senior', 'Senior'],
    'Income': [25000, 30000, 40000, 45000, 50000, 55000]
}

df = pd.DataFrame(data)

grouped_stats = df.groupby('Age_Group')['Income'].agg(
    ['mean', 'median', 'min', 'max', 'std']
)

print("Summary Statistics:")
print(grouped_stats)

income_list_by_group = df.groupby('Age_Group')['Income'].apply(list)

print("\nIncome List by Age Group:")
print(income_list_by_group)