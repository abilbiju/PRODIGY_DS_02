import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('task2_data/train.csv')

# Display basic information about the dataset
print("Basic Information:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())

# Handle missing values
# Fill missing 'Age' values with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most common port
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to a large number of missing values
data.drop(columns=['Cabin'], inplace=True)

# Drop rows with any remaining missing values
data.dropna(inplace=True)

# Display the cleaned data information
print("\nCleaned Data Information:")
print(data.info())

# Exploratory Data Analysis (EDA)
# Plot the distribution of the 'Survived' variable
plt.figure(figsize=(8, 6))
sns.countplot(data['Survived'])
plt.title('Distribution of Survival')
plt.show()

# Plot the distribution of 'Age'
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Plot the relationship between 'Pclass' and 'Survived'
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Passenger Class')
plt.show()

# Plot the relationship between 'Sex' and 'Survived'
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival by Sex')
plt.show()

# Plot the relationship between 'Age' and 'Survived'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Age', data=data)
plt.title('Survival by Age')
plt.show()

# Plot the relationship between 'Fare' and 'Survived'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Survival by Fare')
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
numeric_data = data.select_dtypes(include=[float, int])  # Select only numeric columns
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()