import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")  # Not used in EDA, but useful for predictions

# Display basic information
print("Train Data Info:")
print(train_data.info())

# Checking missing values
print("\nMissing Values in Train Data:\n", train_data.isnull().sum())

# Filling missing Age values with median age
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])


# Dropping Cabin column as it has too many missing values
train_data.drop(columns=['Cabin'], inplace=True)

# Checking data after cleaning
print("\nCleaned Train Data Info:")
print(train_data.info())

# Visualizing Survival Rate
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=train_data, palette="coolwarm")
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Survival Rate by Gender
plt.figure(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=train_data, palette="coolwarm")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Probability")
plt.show()

# Survival Rate by Passenger Class
plt.figure(figsize=(6,4))
sns.barplot(x="Pclass", y="Survived", data=train_data, palette="coolwarm")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Probability")
plt.xlabel("Passenger Class")
plt.show()

# Age Distribution of Passengers
plt.figure(figsize=(8,5))
sns.histplot(train_data["Age"], bins=20, kde=True, color="blue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Box Plot of Age vs. Passenger Class
plt.figure(figsize=(8,5))
sns.boxplot(x="Pclass", y="Age", data=train_data, palette="coolwarm")
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.show()
