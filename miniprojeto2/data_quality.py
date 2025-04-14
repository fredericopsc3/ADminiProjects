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