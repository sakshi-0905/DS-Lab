# Q2) Write a Python program to display some basic statistical details like percentile, mean, 
# standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of 
# iris.csv dataset. 
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=columns)

df = df.dropna()

species_list = df['species'].unique()

for sp in species_list:
    print(f"\nStatistics for {sp}:\n")
    
    species_data = df[df['species'] == sp]
    
    print("Mean:\n", species_data.mean(numeric_only=True))
    
    print("\nStandard Deviation:\n", species_data.std(numeric_only=True))
    
    print("\nPercentiles:")
    print(species_data.quantile([0.25, 0.5, 0.75], numeric_only=True))