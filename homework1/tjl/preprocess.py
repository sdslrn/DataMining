import pandas as pd

# Define column names based on the Boston Housing dataset attributes
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Load the data from the file
data = pd.read_csv('../data/housing.data.txt', delim_whitespace=True, header=None)

# Assign column names to the dataframe
data.columns = column_names

# Save the dataframe to a CSV file
data.to_csv('boston_housing.csv', index=False)

# Print the dataframe
print(data)

plt.boxplot(X[:,0],showmeans=True,meanline=True)
plt.show()
