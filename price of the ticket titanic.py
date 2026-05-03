# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load inbuilt Titanic dataset
df = sns.load_dataset('titanic')

# Display first few rows
print(df.head())

# ----------------------------------------
# 1. Find patterns in data (Seaborn plots)
# ----------------------------------------

# Survival count
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival based on gender
sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival based on Gender")
plt.show()

# Survival based on passenger class
sns.countplot(x='survived', hue='pclass', data=df)
plt.title("Survival based on Passenger Class")
plt.show()

# ----------------------------------------
# 2. Histogram of Fare (Ticket Price)
# ----------------------------------------

plt.hist(df['fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()