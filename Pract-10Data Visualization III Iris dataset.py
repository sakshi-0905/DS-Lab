# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target (species)
df['species'] = iris.target

# Map numeric target to names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# -------------------------------
# 1. Features and their types
# -------------------------------
print("Features and their types:\n")
print(df.dtypes)

# -------------------------------
# 2. Histogram for each feature
# -------------------------------
df.hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# -------------------------------
# 3. Boxplot for each feature
# -------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df.iloc[:, :-1])  # exclude species column
plt.title("Boxplot of Iris Features")
plt.show()

# -------------------------------
# 4. Boxplot with species (better analysis)
# -------------------------------
for col in df.columns[:-1]:
    plt.figure()
    sns.boxplot(x='species', y=col, data=df)
    plt.title(f"{col} vs Species")
    plt.show()