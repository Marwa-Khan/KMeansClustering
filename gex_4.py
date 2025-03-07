import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

class KMeansClustering:
    def __init__(self, file_path=None):
        
        """Add the necessary logic as instructed in the Exercise instruction document"""
       
        if file_path:
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.DataFrame() 

        # self.df = pd.read_csv(file_path)
        # print(self.df.head())
        # X_scaled=""

    def preprocess_data(self):
        
        """Add the necessary logic as instructed in the Exercise instruction document"""
        # Drop rows with missing values and then duplicates
        df = self.df.dropna()
        df = self.df.drop_duplicates()
        print(f"preprocessing {df.shape}")

        # First, select only the numerical columns
        numerical_columns = self.df.select_dtypes(include='number').columns

        #Then iterate over every column, define its 1st and 99th percentiles, and then clip the column to these values.
        for col in numerical_columns:
            lower_bound = self.df[col].quantile(0.01) # 'quantile' returns the value at the given quantile
            upper_bound = self.df[col].quantile(0.99)

            # 'clip' limits the values in the column to the values between lower and upper bounds
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            print(f"{col} outliers removed.")

        print("Outliers capped at 1st and 99th percentiles.")

        print(df.describe())

        # Normalize the data for better clustering results
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(df)

        #The following line is just for demonstration purposes only to see what the statistics of the transformed data look like.
        print(pd.DataFrame(self.X_scaled).describe())
        

    def plot_elbow_curve(self):
        
        """Add the logic to plot the elbow curve."""
        inertia = [] # Sum of squared distances of data points to their closest cluster centroid

        # The following defines a range of number of clusters that we iterate over to minimize Inertia
        # Typically, tha range is from 1 to 10, but it can be adjusted based on the problem
        # # We use (1,11) to include 10 in the range. 
        k_values = range(1, 11) 

        for k in k_values:
            # The following line creates a KMeans model with k clusters. 
            # n_init=10 means that the algorithm will run 10 times
            # Each time, it will try to select a different centroid to minimize Inertia.
            # You can have a higher number but that will take longer to run.
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

            # The following line fits the model to the scaled data
            kmeans.fit(self.X_scaled)

            # The following line appends the inertia of the model to the inertia list.
            # This will be used to plot the Elbow Curve
            inertia.append(kmeans.inertia_)

        # Plot the Elbow Curve
        # plt.plot(k_values, inertia, marker='o', linestyle='-')
        # plt.xlabel('Number of Clusters (k)')
        # plt.ylabel('Inertia (WCSS)')
        # plt.title('Elbow Method for Optimal k')
        # plt.show()
        print(f"Shape of X_scaled: {self.X_scaled.shape}")
        

    def apply_kmeans(self):
        """Add the necessary logic as instructed in the Exercise instruction document"""
        optimal_k = 3
        # Step 1: Create and fit the KMeans model
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X_scaled)  # Fit the model and predict cluster labels

        # Step 2: Add the cluster labels as a new column in the original DataFrame
        self.df['Cluster'] = self.labels  # Add 'Cluster' column to the original DataFrame
        print(self.df.describe)

        # Step 3: Store the cluster centers
        self.centers = self.kmeans.cluster_centers_

        print(f"KMeans clustering applied successfully with {optimal_k} clusters.")
        print("Cluster labels have been added to the DataFrame.")



    def evaluate_clustering(self):
        """Add the necessary logic as instructed in the Exercise instruction document"""
        silhouette = silhouette_score(self.X_scaled, self.df['Cluster'])
        davies_bouldin = davies_bouldin_score(self.X_scaled, self.df['Cluster'])
        ch_index = calinski_harabasz_score(self.X_scaled, self.df['Cluster'])
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        print(f"Calinski-Harabasz Index: {ch_index}")

        return silhouette, davies_bouldin, ch_index

    def visualize_clusters(self):
        """Add the necessary logic as instructed in the Exercise instruction document"""
        # Reduce the dimensionality of the data to 2 dimensions using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        # Create a DataFrame with the PCA components
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

        # Concatenate the cluster labels to the DataFrame
        pca_df['Cluster'] = self.df['Cluster']

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='tab10')
        plt.title('KMeans Clustering')
        plt.show()

        print("Clusters visualized successfully.")

# Example usage
if __name__ == "__main__":
    """Add the necessary logic as instructed in the Exercise instruction document"""
    # initializing class object
    file_path= "test_data.csv"
    kmeans=KMeansClustering(file_path)
    kmeans.preprocess_data()
    kmeans.plot_elbow_curve()
    kmeans.apply_kmeans()
    kmeans.evaluate_clustering()
    kmeans.visualize_clusters()

    
