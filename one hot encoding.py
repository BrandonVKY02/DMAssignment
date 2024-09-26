import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv('C:\Users\Brandon VKY\Desktop\Data Mining Assignment\((GAssign) BankLoanApproval.csv')

# Define categorical columns
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Encode categorical data
encoded_data = encoder.fit_transform(df[categorical_columns])

# Create DataFrame with encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Replace 'Yes' and 'No' with 1 and 0 for specified columns
columns_to_replace = ['HasMortgage', 'HasDependents', 'HasCoSigner']
df[columns_to_replace] = df[columns_to_replace].replace({'Yes': 1, 'No': 0})

# Drop original categorical columns and 'LoanID'
columns_to_drop = categorical_columns + ['LoanID']
df.drop(columns_to_drop, axis=1, inplace=True)

# Concatenate original DataFrame with encoded DataFrame
final_data = pd.concat([df, encoded_df], axis=1)

# Save the final data to a CSV file
final_data.to_csv('one_hot_encoded_data.csv',index=False)
