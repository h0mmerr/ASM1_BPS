import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('abc_manufacturing_sales.csv')


#
# // Handling missing values //
#

# Check for missing values in the 'Sales Quantity' column
missing_values_count = df['Sales Quantity'].isnull().sum()
print("Number of missing values in 'Sales Quantity':", missing_values_count)

# Replace missing values with the mean value of the column
mean_sales_quantity = df['Sales Quantity'].mean()
df['Sales Quantity'].fillna(mean_sales_quantity, inplace=True)

# Confirm that missing values have been handled
missing_values_count_after = df['Sales Quantity'].isnull().sum()
print("Number of missing values after handling:", missing_values_count_after)


#
# // Removing duplicate //
#

# Check for duplicate entries in the dataset
duplicate_rows = df[df.duplicated()]

# Print the duplicate entries
print("Duplicate Entries:")
print(duplicate_rows)

# Remove duplicate entries from the dataset
df.drop_duplicates(inplace=True)

# Verify that duplicate entries have been removed
print("Number of entries after removing duplicates:", len(df))


#
# // Handling inconsistent data format // 
#

# Check unique formats in the 'Revenue' column
unique_formats = df['Revenue'].unique()
print("Unique formats in 'Revenue' column:", unique_formats)

# Remove non-numeric characters and convert to float
df['Revenue'] = df['Revenue'].replace('[\$,]', '', regex=True).astype(float)

# Verify the standardized formatting
print("Data types after formatting 'Revenue' column:")
print(df.dtypes)


#
# // Handling categorical data
#

# One-hot encoding using pandas
df_encoded = pd.get_dummies(df, columns=['Product Category'])

# Label encoding using scikit-learn
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Product Category'] = label_encoder.fit_transform(df['Product Category'])


#
# // Scaling //
#

# Extract numerical features for scaling
numerical_features = df[['Sales Quantity', 'Revenue']]

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical features
scaled_features = scaler.fit_transform(numerical_features)

# Replace original numerical features with scaled features
df[['Sales Quantity', 'Revenue']] = scaled_features
