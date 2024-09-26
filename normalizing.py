import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('C:\Users\Brandon VKY\Desktop\Data Mining Assignment\one_hot_encoded_data.csv')

df.columns
result:
Index(['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',   'HasMortgage',
       'HasDependents', 'HasCoSigner', 'Default', 'Education_Bachelor's',
       'Education_High School', 'Education_Master's', 'Education_PhD',
       'EmploymentType_Full-time', 'EmploymentType_Part-time',
       'EmploymentType_Self-employed', 'EmploymentType_Unemployed',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'LoanPurpose_Auto', 'LoanPurpose_Business',
       'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other'],
      dtype='object')

# Specify columns to normalize
columns_to_normalize = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm', 'DTIRatio']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Create a new DataFrame to store the normalized data
normalized_df = df.copy()

# Normalize specified columns
normalized_df[columns_to_normalize] = scaler.fit_transform(normalized_df[columns_to_normalize])

# Save the normalized DataFrame to a new CSV file
normalized_df.to_csv('normalized_data.csv', index=False)
