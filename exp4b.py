import numpy as np
import csv

# Function to check if a value can be converted to an integer
def is_valid_integer(value):
    try:
        int(value)  # Attempt to convert to int
        return True
    except ValueError:
        return False

# Read data from CSV file and handle non-numeric values
data = []
with open('exp4b.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header row
    for row in reader:
        # Filter out non-numeric values and convert to integers
        numeric_row = [int(value) if is_valid_integer(value) else 0 for value in row]
        data.append(numeric_row)

# Convert filtered data to numpy array with dtype=int
data_array = np.array(data[1:], dtype=int)  # Skip header row

# Extract observed frequencies
observed = data_array[:2, :2]

# Calculate row and column totals
row_totals = np.sum(observed, axis=1)
col_totals = np.sum(observed, axis=0)

# Calculate total observations (avoid division by zero)
total_obs = np.sum(observed)
if total_obs == 0:
    total_obs = 1  # Set to a non-zero value to avoid division by zero

# Calculate expected frequencies
expected = np.outer(row_totals, col_totals) / total_obs

# Calculate the chi-square statistic (replace NaN with 0)
chi_square = np.sum((observed - expected)**2 / np.nan_to_num(expected))

# Calculate degrees of freedom
df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

# Print the results for Chi-Square Test
print("Chi-Square Test:")
print("Observed Frequencies:")
print(observed)
print("\nExpected Frequencies:")
print(expected)
print("\nChi-square Statistic:", chi_square)
print("Degrees of Freedom:", df)

# Define the variables for correlation analysis
right_handed = data_array[:, 0]
left_handed = data_array[:, 1]

# Calculate means
mean_right = np.mean(right_handed)
mean_left = np.mean(left_handed)

# Calculate covariance (replace NaN with 0)
covariance = np.sum((right_handed - mean_right) * (left_handed - mean_left)) / len(right_handed)

# Calculate standard deviations (replace NaN with 0)
std_dev_right = np.std(right_handed, ddof=1)
std_dev_left = np.std(left_handed, ddof=1)

# Calculate correlation coefficient (replace NaN with 0)
correlation_coefficient = covariance / (np.nan_to_num(std_dev_right) * np.nan_to_num(std_dev_left))

# Print the results for Correlation Analysis (Pearson)
print("\nCorrelation Analysis (Pearson):")
print("Correlation Coefficient:", correlation_coefficient)
