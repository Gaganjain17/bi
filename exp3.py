import pandas as pd
import numpy as np

# Read
df = pd.read_csv('exp3.csv')

# Display the original dataset
print("Original Dataset:")
print(df)
print("\n")

df_cleaned = df.dropna()

# Display the cleaned dataset
print("Cleaned Dataset (Removed Rows with Missing Values):")
print(df_cleaned)
print("\n")

# b. Handling Missing Data: Filling missing values with mean
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_filled = df.copy()
df_filled[numeric_columns] = df_filled[numeric_columns].fillna(df_filled[numeric_columns].mean())

# Display the dataset with missing values filled
print("Dataset with Missing Values Filled (Using Mean):")
print(df_filled)
print("\n")

# c. Data Transformation: Adding a new column (e.g., Adjusted Salary)
df_filled['Adjusted Salary'] = df_filled['Salary'] * 1.1

# Display the transformed dataset
print("Transformed Dataset (Added Adjusted Salary Column):")
print(df_filled)
print("\n")

# Save the cleaned and transformed DataFrames to CSV files
df_cleaned.to_csv('cleaned_data_experiment.csv', index=False)
df_filled.to_csv('transformed_data_experiment.csv', index=False)

