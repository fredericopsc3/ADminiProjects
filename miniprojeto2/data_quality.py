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

# Count missing values per column
print("\nMissing Values:")
print(cancer_df.isnull().sum())

# **2. Handling Missing Values**
# Determine the percentage of missing values in each column
missing_percentage = (cancer_df.isnull().sum() / len(cancer_df)) * 100
print("\nPercentage of Missing Values in each column:")
print(missing_percentage)

# Impute missing values
cancer_df['Cause_of_Death'] = cancer_df['Cause_of_Death'].fillna('Alive')

# **3. Identifying Outliers**
#Use the Interquartile Range (IQR) method to detect outliers
for column in ['Age', 'Weight', 'Height']:
    Q1 = cancer_df[column].quantile(0.25)
    Q3 = cancer_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = cancer_df[(cancer_df[column] < lower_bound) | (cancer_df[column] > upper_bound)]
    print(f"{column}:")
    print(len(outliers))
    
# Use a box plot to visualize outliers
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=cancer_df[column])
    plt.title("Box Plot for Outlier Detection")
    plt.show()
    
# **4. Handling Outliers**
#Remove outliers
    outliers_filtered = cancer_df[(cancer_df[column] <= lower_bound) & (cancer_df[column] >= upper_bound)]
    print("\nAfter Handling Outliers:")
    print(f"{column}:")
    print(len(outliers_filtered))

cancer_df.to_csv('cancer_dataset_cleaned.csv', index=False)