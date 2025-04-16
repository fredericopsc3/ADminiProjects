import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

categorical_columns = ['gender', 'nationality', 'cancer_type', 'outcome', 'treatment_type']
numerical_columns = ['age', 'weight', 'height']

for col in categorical_columns:
    if col in cancer_df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=cancer_df, x=col, order=cancer_df[col].value_counts().index)
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

for col in numerical_columns:
    if col in cancer_df.columns:
        # Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(cancer_df[col], kde=False, bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # KDE Plot
        plt.figure(figsize=(8, 4))
        sns.kdeplot(cancer_df[col], fill=True)
        plt.title(f'Density Curve (KDE) of {col}')
        plt.xlabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Boxplot
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=cancer_df[col])
        plt.title(f'Boxplot of {col}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Violin Plot
        plt.figure(figsize=(8, 4))
        sns.violinplot(x=cancer_df[col])
        plt.title(f'Violin Plot of {col}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if 'gender' in cancer_df.columns and 'age' in cancer_df.columns:
    # Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='gender', y='age', data=cancer_df)
    plt.title(f"Boxplot of {'age'} by {'gender'}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='gender', y='age', data=cancer_df)
    plt.title(f"Violin plot of {'age'} by {'gender'}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Stripplot
    plt.figure(figsize=(10, 5))
    sns.stripplot(x='gender', y='age', data=cancer_df, jitter=True, alpha=0.5)
    plt.title(f"Stripplot of {'age'} by {'gender'}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if 'weight' in cancer_df.columns and 'height' in cancer_df.columns:
    # Scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cancer_df, x='weight', y='height', hue=None)
    plt.title(f"Scatterplot: {'weight'} vs {'height'}" + (f" by {None}" if None else ""))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Regression line
    plt.figure(figsize=(8, 6))
    sns.regplot(data=cancer_df, x='weight', y='height', scatter_kws={'alpha': 0.3})
    plt.title(f"Regression Line: {'weight'} vs {'height'}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Correlation coefficient
    correlation = cancer_df[['weight', 'height']].corr().iloc[0, 1]
    print(f"Correlation coefficient between {'weight'} and {'height'}: {correlation:.2f}")

# Histogram using FacetGrid
g = sns.FacetGrid(cancer_df, col='gender', height=4, aspect=1.2)
g.map(sns.histplot, 'age', bins=20)
g.set_axis_labels('age', "Count")
g.figure.suptitle(f"Histogram of {'age'.title()} by {'gender'.title()}", y=1.05)
plt.show()

# Boxplot by two categories
sns.catplot(data=cancer_df, x="emirate", y="weight", hue=None, kind='box', height=5, aspect=1.5)
plt.title(f"Boxplot of {"weight".title()} by {"emirate".title()}" + (f" and {None.title()}" if None else ""))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatterplot, faceted by Region (Emirate) facet_scatterplot("age", "height", col_by="emirate")
sns.relplot(data=cancer_df, x="age", y="height", col="emirate", kind='scatter', height=4, aspect=1)
plt.suptitle(f"{"height".title()} vs {"age".title()} by {"emirate".title()}", y=1.03)
plt.tight_layout()
plt.show()