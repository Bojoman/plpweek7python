# Data Analysis Project - Exploring the Iris Dataset

### Objective:
This project aims to explore and analyze the Iris dataset to uncover patterns in the data using pandas, matplotlib, and seaborn.

### Libraries used:
- **pandas** for data manipulation
- **matplotlib** and **seaborn** for data visualization
- **scikit-learn** for loading the dataset

---

## 1. Data Loading and Exploration

### Load and Inspect the Data

```python
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display the first 5 rows
df.head()
# Check for missing values
df.isnull().sum()

# Check data types
df.dtypes

# In this case, there are no missing values, but if there were:
# Drop rows with missing values
df_cleaned = df.dropna()

# Or fill missing values with the mean of the column
df_filled = df.fillna(df.mean())
# Compute basic statistics of the numerical columns
df.describe()
# Group by species and calculate the mean
grouped = df.groupby('species').mean()
grouped
import matplotlib.pyplot as plt

# Line chart for sepal length (if there were a time-based column)
df['sepal length (cm)'].plot(kind='line')
plt.title('Sepal Length Over Time')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.show()

import seaborn as sns

# Bar chart for comparing average petal length per species
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()
# Histogram for sepal length distribution
plt.hist(df['sepal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for sepal length vs petal length
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], color='purple')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()


