import pandas as pd
import matplotlib.pyplot as plt


file_path = 'exp4a.csv'  
logistics_data = pd.read_csv(file_path)

print("Logistics Data:")
print(logistics_data)


#fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plt.subplot(2,2,1)
#  Bar Chart for Shipment Quantities
plt.bar(logistics_data['Shipment_Number'], logistics_data['Shipment_Quantity'], color='skyblue')
plt.title('Bar Chart - Shipment Quantities')
plt.xlabel('Shipment Number')
plt.ylabel('Quantity')

#  Histogram for Delivery Times
plt.subplot(2,2,2)
plt.hist(logistics_data['Delivery_Time'], bins=10, color='lightcoral', edgecolor='black')
plt.title('Histogram - Delivery Times')
plt.xlabel('Delivery Time (days)')
plt.ylabel('Frequency')

#  Scatter Plot for Delivery Times vs. Transportation Costs
plt.subplot(2,2,3)
plt.scatter(logistics_data['Delivery_Time'], logistics_data['Transportation_Cost'], color='gold', alpha=0.7)
plt.title('Scatter Plot - Delivery Times vs. Transportation Costs')
plt.xlabel('Delivery Time (days)')
plt.ylabel('Transportation Cost ($)')

#  Box Plot for Shipment Quantities
plt.subplot(2,2,4)
plt.boxplot(logistics_data['Shipment_Quantity'])
plt.title('Box Plot - Shipment Quantities')
plt.ylabel('Quantity')


plt.tight_layout()
plt.show()
