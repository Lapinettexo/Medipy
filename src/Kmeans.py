import os
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA


# Step 1: Loading the data
def load_data_from_json(folder_path):
    features = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            
            base_name = os.path.splitext(file_name)[0]  # Removing .json extension
            label = '_'.join(base_name.split('_')[:2])  # Taking only the first two words

            with open(file_path, 'r') as file:
                data = json.load(file)
                
                for image_name, parts_data in data.items():
                    # putting all the parts of the image frequency vectors into one feature vector
                    image_features = []
                    for part_key, freq_vector in parts_data.items():
                        image_features.extend(freq_vector)
                    
                    features.append(image_features)
                    labels.append(label)

    return features, labels

folder_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Frequencies//Data"
X, y = load_data_from_json(folder_path)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=50)  # Reduce dimensions to 50 components
X_pca = pca.fit_transform(X_scaled)

# Explained variance plot
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()


# Function to find optimal K
def find_optimal_k(X, max_k=10):
    distortions = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, distortions, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.show()

    # Return optimal K based on highest silhouette score
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal K (based on Silhouette Score): {optimal_k}")
    return optimal_k

# Find the optimal K using PCA-transformed data
optimal_k = find_optimal_k(X_pca)

# Apply K-Means with the optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_pca)

# Add cluster labels to your data
cluster_labels = kmeans.labels_

# Evaluate clustering results
silhouette = silhouette_score(X_pca, cluster_labels)
print("Silhouette Score for final clustering:", silhouette)

# If you have ground truth labels, evaluate clusters
ari = adjusted_rand_score(y, cluster_labels)
print("Adjusted Rand Index (against true labels):", ari)