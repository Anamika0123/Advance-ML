#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# # DATA EXPOLATION

# In[2]:


df  = pd.read_csv("heart_disease_uci.csv")


# In[3]:


df


# In[4]:


df .head()


# In[5]:


df.isnull()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[9]:


categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['number']).columns
print("Categorical Features:\n", categorical_features)
print("Numerical Features:\n", numerical_features)


# # Handling Missing Data:

# In[10]:


df['trestbps'].fillna(df['trestbps'].mean(), inplace=True)


# In[11]:


df


# In[14]:


# Impute missing values in 'Ca' and 'Thal' using the most frequent value (mode)
df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)

# Verify the changes
print(df[['ca', 'thal']].isnull().sum())


# # Feature creation

# In[15]:


#Create new features based on existing columns to add more information:


# In[16]:


# Define age groups
bins = [0, 40, 60, 120]  # Age categories: <40, 40-60, >60
labels = ['<40', '40-60', '>60']

# Create a new column for age groups
df['Age Group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Display the updated DataFrame
print(df[['age', 'Age Group']].head())


# In[19]:


# Define a function to categorize cholesterol levels
def categorize_cholesterol(chol):
    if chol < 200:
        return 'Low'
    elif 200 <= chol <= 239:
        return 'Normal'
    else:
        return 'High'
# Apply the function to the 'chol' column (replace 'chol' with the actual column name if different)
df['Cholesterol_Category'] = df['chol'].apply(categorize_cholesterol)
print(df[['chol', 'Cholesterol_Category']].head())


# In[20]:


# Define criteria for high risk
high_cholesterol_threshold = 240  # mg/dl
high_blood_pressure_threshold = 140  # mm Hg
age_threshold = 60  # years

# Create a binary feature 'high_risk'
df['high_risk'] = ((df['chol'] > high_cholesterol_threshold) | 
                     (df['trestbps'] > high_blood_pressure_threshold) | 
                     (df['age'] > age_threshold)).astype(int)

# Display the updated DataFrame with the new 'high_risk' feature
print(df[['age', 'chol', 'trestbps', 'high_risk']])


# # Feature Transformation:

# In[26]:


# **Label Encoding**
label_columns = ['sex', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'dataset']
label_encoders = {}

for column in label_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store the encoder for future use

print("\nData after Label Encoding:")
print(df.head())

# **One-Hot Encoding**
# Select categorical columns for One-Hot Encoding
one_hot_columns = ['cp', 'dataset']  # Example columns to one-hot encode

df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

print("\nData after One-Hot Encoding:")
print(df.head())


# In[41]:


from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# Display original DataFrame shape and columns
print("Original DataFrame shape:", df.shape)
print("Columns in DataFrame:", df.columns.tolist())

# Display the first few rows of the dataset
print("Original Data:")
print(df.head())

# **Label Encoding for 'sex' and 'thal'**
label_encoder = LabelEncoder()

# Encode 'sex' (Male=1, Female=0)
df['sex'] = label_encoder.fit_transform(df['sex'])

# Check if 'thal' exists before encoding
if 'thal' in df.columns:
    # Encode 'thal' (assuming thal has values like 'normal', 'fixed defect', 'reversable defect')
    df['thal'] = label_encoder.fit_transform(df['thal'])
else:
    print("Column 'thal' not found in DataFrame.")

# **One-Hot Encoding for 'cp' (chest pain type)**
if 'cp' in df.columns:
    df = pd.get_dummies(df, columns=['cp'], drop_first=True)
else:
    print("Column 'cp' not found in DataFrame.")

# **Creating AgeGroup Feature**
# Define age groups based on age ranges
bins = [0, 30, 40, 50, 60, 70, 80]
labels = ['0-30', '31-40', '41-50', '51-60', '61-70', '71-80']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)

# One-Hot Encoding for AgeGroup
df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)

# Display the updated DataFrame with encoded features
print("\ndf after Encoding:")
print(df.head())


# In[33]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Display the first few rows of the dataset
print("Original Data:")
print(df[['chol', 'trestbps', 'thalch']].head())

# **Using MinMaxScaler**
min_max_scaler = MinMaxScaler()

# Normalize 'chol', 'trestbps', and 'thalch' using MinMaxScaler
df[['chol', 'trestbps', 'thalch']] = min_max_scaler.fit_transform(df[['chol', 'trestbps', 'thalch']])

print("\nData after MinMax Scaling:")
print(df[['chol', 'trestbps', 'thalch']].head())

# **Using StandardScaler**
standard_scaler = StandardScaler()

# Normalize 'chol', 'trestbps', and 'thalch' using StandardScaler
df[['chol', 'trestbps', 'thalch']] = standard_scaler.fit_transform(df[['chol', 'trestbps', 'thalch']])

print("\nData after Standard Scaling:")
print(df[['chol', 'trestbps', 'thalch']].head())


# # Feature Interaction:

# In[34]:


#Create interaction features by combining two or more features:


# In[35]:


# Display the first few rows of the dataset before creating the interaction feature
print("Original Data:")
print(df[['trestbps', 'chol']].head())

# Create an interaction feature by multiplying 'trestbps' and 'chol'
df['BP_Chol_Interaction'] = df['trestbps'] * df['chol']

# Display the updated DataFrame with the new interaction feature
print("\nData after adding BP-Chol Interaction feature:")
print(df[['trestbps', 'chol', 'BP_Chol_Interaction']].head())


# In[36]:


# Display the first few rows of the dataset to understand its structure
print("Original Data:")
print(df[['exang', 'thalch']].head())

# Define the threshold for thalach
threshold = 100

# Create a binary feature 'high_risk'
df['high_risk'] = ((df['exang'] == 'TRUE') & (df['thalch'] < threshold)).astype(int)

# Display the updated DataFrame with the new 'high_risk' feature
print("\nData after adding high_risk feature:")
print(df[['exang', 'thalch', 'high_risk']].head())


# # Feature Selection:

# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Option 1: Drop rows with missing values (if any)
df = df.dropna()

# Option 2: Alternatively, fill missing values (e.g., with the mean for numerical columns)
# df.fillna(df.mean(), inplace=True)

# Preprocessing: Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['sex', 'dataset', 'cp', 'restecg', 'exang', 'slope', 'thal']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Define features and target variable
X = df.drop(columns=['num'])  # Features (all columns except target)
y = df['num']  # Target variable (presence of heart disease)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
X_train[['chol', 'trestbps', 'thalch']] = scaler.fit_transform(X_train[['chol', 'trestbps', 'thalch']])
X_test[['chol', 'trestbps', 'thalch']] = scaler.transform(X_test[['chol', 'trestbps', 'thalch']])

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# Display the feature importance DataFrame
print(feature_importance_df)


# In[40]:


from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# Display original DataFrame shape
print("Original DataFrame shape:", df.shape)

# Step 1: Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['sex', 'dataset', 'cp', 'restecg', 'exang', 'slope', 'thal']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Step 2: Identify and drop low variance features
# Set a variance threshold (e.g., 0.01)
var_threshold = VarianceThreshold(threshold=0.01)
var_threshold.fit(df)

# Get the columns that have variance above the threshold
features_with_variance = df.columns[var_threshold.get_support()]

# Create a new DataFrame with only high variance features
df_high_variance = df[features_with_variance]

# Display shape after dropping low variance features
print("Shape after dropping low variance features:", df_high_variance.shape)

# Step 3: Identify and drop highly correlated features
# Calculate the correlation matrix
correlation_matrix = df_high_variance.corr()

# Set up a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Find features with correlation greater than 0.8
threshold = 0.8
to_drop = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            to_drop.add(colname)

# Drop highly correlated features from the DataFrame
df_final = df_high_variance.drop(columns=to_drop)

# Display final DataFrame shape and dropped columns
print("Final DataFrame shape:", df_final.shape)
print("Dropped columns due to high correlation:", to_drop)


# In[ ]:




