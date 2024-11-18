import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load the data
data = pd.read_csv('Mall_Customers.csv')

# Step 2: Inspect the first few rows to understand the structure
print(data.head())

# Step 3: Preprocessing - Select relevant columns for clustering
data = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Normalize the data (standardize it to have zero mean and unit variance)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 5: Apply K-means clustering (let's assume 5 clusters for now)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Step 6: Add the cluster labels to the original dataframe
data['Cluster'] = clusters

# Step 7: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()

# Step 8: Print the cluster centers (in original scale)
cluster_centers = kmeans.cluster_centers_  # These are in the scaled space
cluster_centers_original = scaler.inverse_transform(cluster_centers)  # Convert back to original scale
print("Cluster centers (in original scale):")
print(cluster_centers_original)

# Step 9: Optionally, save the results to a new CSV with the cluster labels
data.to_csv('Mall_Customers_Clustered.csv', index=False)
