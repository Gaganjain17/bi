import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Create a sample dataset of transactions
data = {
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [['bread', 'milk'], ['bread', 'diaper', 'beer', 'egg'], 
              ['milk', 'diaper', 'beer', 'cola'], ['bread', 'milk', 'diaper', 'beer'],
              ['bread', 'milk', 'diaper', 'cola']]
}

df = pd.DataFrame(data)

# Convert the 'Items' column into a suitable format for Apriori algorithm
df['Items'] = df['Items'].apply(lambda x: ','.join(x))  # Convert list to string for demonstration

# Apply the Apriori algorithm to find frequent itemsets with minimum support threshold
frequent_itemsets = apriori(df['Items'].str.get_dummies(sep=','), min_support=0.4, use_colnames=True)

# Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Calculate support and confidence for the association rules
rules['support'] = rules['support'] * len(df)  # Calculate absolute support
rules['confidence'] = rules['confidence'] * 100  # Convert confidence to percentage

# Print the association rules with support and confidence
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

# Example of filtering rules based on support and confidence thresholds
filtered_rules = rules[(rules['support'] >= 2) & (rules['confidence'] >= 70)]

print("\nFiltered Association Rules:")
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence']])
