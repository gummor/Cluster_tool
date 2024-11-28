import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import (
    KMeans, 
    DBSCAN, 
    Birch, 
    MiniBatchKMeans, 
    AgglomerativeClustering, 
    OPTICS, 
    MeanShift, 
    SpectralClustering
)
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score, 
    adjusted_rand_score, 
    calinski_harabasz_score
)

data = pd.read_csv(loren)

scaler = StandardScaler()
data['N_normalized'] = scaler.fit_transform(data[['N']])
data['Iq1_normalized'] = scaler.fit_transform(data[['Iq1']])
data['Iq3_normalized'] = scaler.fit_transform(data[['Iq3']])
data['Copeptin_normalized'] = scaler.fit_transform(data[['Copeptin']])

data['cv'] = (data['Iq3_normalized'] - data['Iq1_normalized']) / data['Copeptin_normalized']

scaler = StandardScaler()
selected_columns = data[['cv', 'Iq3_normalized', 'Iq1_normalized', 'Copeptin_normalized', 'N_normalized']]

pca = PCA(n_components=4)
pca_result = pca.fit_transform(selected_columns)

pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Define cluster names
cluster_names = {0: 'Controle', 1: 'Preeclampsia'}

# Dictionary of clustering models
clustering_models = {
    "KMeans": KMeans(n_clusters=2, random_state=0),
    "GaussianMixture": GaussianMixture(n_components=2, random_state=0),
    "SpectralClustering": SpectralClustering(n_clusters=2, random_state=0),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),  # Adjust parameters as necessary
    "Birch": Birch(n_clusters=2),
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=2, random_state=0),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=2),
    "OPTICS": OPTICS(min_samples=5),  # Adjust parameters as necessary
    "MeanShift": MeanShift(),
}

# Step 4: Train and Evaluate Clustering Models
cluster_results = {}
evaluation_metrics = {}

# Assuming you have true_labels for adjusted_rand_score
true_labels = data['Group']  # Defina aqui seus rótulos verdadeiros
features = data[['cv', 'Iq3_normalized', 'Iq1_normalized', 'Copeptin_normalized', 'N_normalized']]  # Use a list to select columns
        
# Step 4: Train and Evaluate Clustering Models
for model_name, model in clustering_models.items():
    # Fit the model on the feature data, excluding true labels
    model.fit(features)
    
    # Predict the cluster labels
    cluster_labels = model.labels_ if hasattr(model, 'labels_') else model.predict(features)

    # Store the results
    cluster_results[model_name] = cluster_labels

    # Evaluate the model using adjusted Rand index
    evaluation_metrics[model_name] = adjusted_rand_score(true_labels, cluster_labels)
    
    # Calculate evaluation metrics
    silhouette = silhouette_score(selected_columns, cluster_labels)
    davies_bouldin = davies_bouldin_score(selected_columns, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(selected_columns, cluster_labels)
    
    # Adjusted Rand Score requires true labels
    adjusted_rand = adjusted_rand_score(true_labels, cluster_labels)
    
    # Store the metrics
    evaluation_metrics[model_name] = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "Calinski-Harabasz Score": calinski_harabasz,
        "Adjusted Rand Score": adjusted_rand
    }

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(evaluation_metrics).T  # Transpose for better readability

# Display the metrics
print(metrics_df)