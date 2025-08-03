
import pandas as pd

# Create a sample dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
    'Age': [25, 30, 35, 40, 45],
    'Department': ['HR', 'IT', 'Sales', 'IT', 'HR'],
    'Salary': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)

# Save the dataframe to an Excel file
df.to_excel("test_data.xlsx", index=False)
