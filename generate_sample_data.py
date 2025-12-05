"""
Generate sample CSV with various data quality issues for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_rows = 1000

data = {
    # Numeric columns with outliers
    'customer_id': range(1, n_rows + 1),
    'age': np.random.randint(18, 80, n_rows),
    'income': np.random.normal(50000, 15000, n_rows),
    'credit_score': np.random.randint(300, 850, n_rows),
    
    # Categorical columns
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_rows, p=[0.48, 0.48, 0.04]),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_rows),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows, p=[0.3, 0.4, 0.2, 0.1]),
    
    # Boolean column
    'is_employed': np.random.choice([True, False], n_rows, p=[0.85, 0.15]),
    
    # Datetime column
    'signup_date': [datetime(2020, 1, 1) + timedelta(days=int(x)) for x in np.random.randint(0, 1095, n_rows)],
    
    # Column with many missing values
    'optional_field': np.random.choice([np.nan, 'Value A', 'Value B', 'Value C'], n_rows, p=[0.7, 0.1, 0.1, 0.1]),
    
    # Column with some missing values
    'phone_number': np.random.choice([np.nan, '123-456-7890', '987-654-3210', '555-123-4567'], n_rows, p=[0.15, 0.3, 0.3, 0.25]),
    
    # Constant column (should be dropped)
    'constant_value': ['CONSTANT'] * n_rows,
    
    # High cardinality text column
    'email': [f'user{i}@example.com' for i in range(n_rows)],
}

df = pd.DataFrame(data)

# Add some outliers to income
outlier_indices = np.random.choice(n_rows, size=20, replace=False)
df.loc[outlier_indices, 'income'] = np.random.uniform(150000, 300000, 20)

# Add negative values to credit_score (data quality issue)
bad_indices = np.random.choice(n_rows, size=5, replace=False)
df.loc[bad_indices, 'credit_score'] = -999

# Add duplicates
duplicate_rows = df.sample(n=10, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Add row with all missing values
df.loc[len(df)] = [np.nan] * len(df.columns)

# Add spaces to column names (should be standardized)
df.columns = [
    'Customer ID',
    'Age',
    'Annual Income',
    'Credit Score',
    'Gender',
    'City',
    'Education Level',
    'Is Employed',
    'Signup Date',
    'Optional Field',
    'Phone Number',
    'Constant Value',
    'Email Address'
]

# Save to CSV
output_path = 'd:/Project/ML/EDA-Engine/sample_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Sample dataset created: {output_path}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"   Issues included:")
print(f"   - {len(duplicate_rows)} duplicate rows")
print(f"   - 70% missing values in 'Optional Field'")
print(f"   - 15% missing values in 'Phone Number'")
print(f"   - 20 outliers in 'Annual Income'")
print(f"   - 5 invalid credit scores (-999)")
print(f"   - 1 constant column")
print(f"   - 1 row with all missing values")
print(f"   - Column names with spaces")
