# Coffee Quality Analysis

# Importing the necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Setting up inline plotting for Jupyter Notebook
%matplotlib inline

# Defining the path to the data file
data = 'C:\\Users\\Albandari\\Desktop\\TawakkalnaPlatform.csv'

# Reading the data from the file
df = pd.read_csv(data)

# Displaying the first 5 rows of the data
print("First 5 rows of the data:")
print(df.head())

# Displaying the shape of the data (number of rows and columns)
print("Shape of the data (number of rows and columns):", df.shape)

# Extracting numeric columns
num_col = df._get_numeric_data().columns
print("Numeric columns:", num_col)

# Displaying the number of numeric columns
print("Number of numeric columns:", len(num_col))

# Creating a new DataFrame with numeric data
N = df[num_col]

# Extracting categorical (non-numeric) columns
cat_col = [col for col in df.columns if col not in num_col]

# Displaying the number of categorical columns
print("Number of categorical columns:", len(cat_col))

# Creating a new DataFrame with categorical data
C = df[cat_col]

# Displaying the types of data and their counts
print("Data types and counts:")
print(df.dtypes.value_counts())

# General information about the data
print("General information about the data:")
print(df.info())

# Handling missing values
for name in df.columns:
    x = df[name].isna().sum()  # Counting missing values
    if x > 0 and (df.dtypes[name] == float):
        df[name].fillna(df[name].mean(), inplace=True)  # Fill missing values with column mean
    elif (df.dtypes[name] == object):
        df[name].fillna(df[name].mode()[0], inplace=True)  # Fill missing values with most frequent value

# Displaying total number of missing values after processing
total_missing = sum(df.isnull().sum())
print(f"Total missing values after processing: {total_missing}")

# Displaying descriptive statistics of the data
description = df.describe()
print("Descriptive statistics:")
print(description)

# Calculating the correlation matrix
hig_corr = N.corr()

# Identifying features with high correlation with 'Balance'
hig_corr_features = hig_corr.index[abs(hig_corr["Balance"]) >= 0.5]
print("Features with high correlation to 'Balance':", hig_corr_features)

# Visualizing missing values using a heatmap
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title('Missing Values Heatmap')
plt.show()

# Creating boxplots for numeric features
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot for {col}')
plt.tight_layout()
plt.show()

# Plotting distributions of numeric features
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram for {col}')
plt.tight_layout()
plt.show()

# Using pairplot to visualize relationships between features
sns.pairplot(df[num_col])
plt.title('Pairplot of Numeric Features')
plt.show()

# Plotting distribution of the continent of coffee origin
plt.figure(figsize=(10, 6))
plt.hist(df['Continent.of.Origin'], bins=10, color='blue', edgecolor='darkblue', linewidth=2)  
plt.title('Distribution of Continent of Origin')
plt.xlabel('Continent')
plt.ylabel('Number of Observations')
plt.show()

# Displaying the top 10 most common countries
counts = df['Country.of.Origin'].value_counts()
top_10_counts = counts[:10]

# Creating a pie chart for country distribution
plt.pie(top_10_counts, labels=top_10_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Coffee Producing Countries')
plt.show()

# Plotting histograms for numeric data
N.hist(figsize=(20, 20))

# Plotting regression plots for highly correlated features
plt.figure(figsize=(15, 50))
for i in range(len(hig_corr_features)):
    plt.subplot(12, 3, i + 1)
    sns.regplot(data=df, x=hig_corr_features[i], y='Balance')

# Creating a heatmap for the correlation matrix of highly correlated features
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df[hig_corr_features].corr(), annot=True, linewidths=3, cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# Building a simple machine learning model
# Splitting the data into training and testing sets
X = df[hig_corr_features].drop('Balance', axis=1)
y = df['Balance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions using the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Saving the processed data to a CSV file
df.to_csv('processed_coffee_data.csv', index=False)

# Generating a comprehensive statistical summary
summary_stats = df.describe(include='all')
print("Comprehensive Statistical Summary:\n", summary_stats)
