import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the CSV with explicit NA values control
cancer_df = pd.read_csv('_cancer_dataset_uae.csv', keep_default_na=False, na_values=['', 'N/A'])


# **1. Load and explore the data set**
# Display the first 5 rows
print("Preview of Data:")
print(cancer_df.head())
# Check data structure and types
print("\nDataset Info:")
print(cancer_df.info())

# Display summary statistics
summary_stats = cancer_df.describe()
print("Summary Statistics:")
print(summary_stats)

# **2. Perform Data Cleaning**

# Count unique values in each column (useful for spotting categorical vars)
print("\nUnique Values Per Column:")
print(cancer_df.nunique())

# Count missing values per column
print("\nMissing Values:")
print(cancer_df.isnull().sum())

# Handle missing values
# Assumption: If death-related info is missing, the patient is still alive
cancer_df['Cause_of_Death'] = cancer_df['Cause_of_Death'].fillna('Alive')

# Check and drop duplicates
duplicates_count = cancer_df.duplicated().sum()
print(f"\nDuplicates Found: {duplicates_count}")
cancer_df.drop_duplicates(inplace=True)

# Standardize column names
cancer_df.columns = cancer_df.columns.str.lower().str.replace(" ", "_")

# Convert date columns to datetime
date_columns = ['diagnosis_date', 'treatment_start_date', 'death_date']
for col in date_columns:
    cancer_df[col] = pd.to_datetime(cancer_df[col], errors='coerce')

# Check for missing values
print("\nMissing Values After Cleaning:")
print(cancer_df.isnull().sum())

# **3. Conduct Exploratory Data Analysis (EDA)**

# Summary statistics for numerical columns
numerical_summary = cancer_df[['age', 'weight', 'height']].describe()
print("Numerical Summary:")
print(numerical_summary)

# Distribution of categorical columns
for col in cancer_df.select_dtypes(include='object').columns:
    if col in ['patient_id', 'primary_physician']:
        continue

    print(f"Value counts for '{col}':")
    
    # Get value counts and percentages
    value_counts = cancer_df[col].value_counts()
    percentages = cancer_df[col].value_counts(normalize=True) * 100

    # Combine counts and percentages
    for category in value_counts.index:
        count = value_counts[category]
        percent = percentages[category]
        print(f"{category}: {count} ({percent:.2f}%)")
    
    print("-" * 40)

# Correlation matrix between numerical variables
correlation_matrix = cancer_df[['age', 'weight', 'height']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Example: Average age by cancer type
avg_age_by_cancer = cancer_df.groupby('cancer_type')['age'].mean().sort_values(ascending=False)
print("\nAverage Age by Cancer Type:")
print(avg_age_by_cancer)

# Example: Outcome distribution by treatment type
outcome_by_treatment = cancer_df.groupby('treatment_type')['outcome'].value_counts().unstack().fillna(0)
print("\nOutcome Distribution by Treatment Type:")
print(outcome_by_treatment)

# **4. Visualize Insights**

# Histogram of a key numerical variable – Age
plt.figure(figsize=(8, 5))
plt.hist(cancer_df['age'], bins=30, edgecolor='black')
plt.title('Distribution of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Histogram of a key numerical variable – Height
plt.figure(figsize=(8, 5))
plt.hist(cancer_df['height'], bins=30, edgecolor='black')
plt.title('Distribution of Patient Heights')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Histogram of a key numerical variable – Weight
plt.figure(figsize=(8, 5))
plt.hist(cancer_df['weight'], bins=30, edgecolor='black')
plt.title('Distribution of Patient Weights')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Bar chart of average age by Cancer Type
avg_age_by_cancer = cancer_df.groupby('cancer_type')['age'].mean().sort_values()
plt.figure(figsize=(10, 6))
avg_age_by_cancer.plot(kind='barh', color='skyblue')
plt.title('Average Age by Cancer Type')
plt.xlabel('Average Age')
plt.ylabel('Cancer Type')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Scatter plot of Weight vs Height
plt.figure(figsize=(8, 5))
sns.scatterplot(data=cancer_df, x='height', y='weight', alpha=0.5)
plt.title('Scatter Plot of Weight vs Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True)
plt.show()

# Box plot – Age by Outcome
plt.figure(figsize=(8, 5))
sns.boxplot(data=cancer_df, x='outcome', y='age')
plt.title('Age Distribution by Treatment Outcome')
plt.xlabel('Outcome')
plt.ylabel('Age')
plt.grid(True)
plt.show()

# Heatmap – Correlation Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cancer_df[['age', 'weight', 'height']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Age, Weight, Height')
plt.show()

cancer_df.to_csv('cancer_dataset_cleaned.csv', index=False)





