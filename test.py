import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from gex_4 import KMeansClustering  

class TestKMeansClustering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.test_file = "test_data.csv"  # Use test.csv as the dataset
        
        cls.clustering = KMeansClustering()

        cls.clustering.df = pd.read_csv(cls.test_file)
        
        cls.clustering.preprocess_data()
        cls.clustering.apply_kmeans()
        
        cls.silhouette_score = cls.clustering.evaluate_clustering()
    
    @patch.object(KMeansClustering, 'plot_elbow_curve')  
    @patch.object(KMeansClustering, 'visualize_clusters')
    def test_clustering_scores(self, mock_plot_elbow, mock_visualize_clusters):
        
        # Expected values 
        expected_silhouette_score = 0.25  
        expected_davies_bouldin_index = 1.5  
        expected_ch_index = 70  

        # Get actual scores
        silhouette, db_index, ch_index = self.clustering.evaluate_clustering()

        # Assert values 
        self.assertGreater(silhouette, expected_silhouette_score, msg="Silhouette Score is not more than 0.25")
        self.assertLess(db_index, expected_davies_bouldin_index, msg="Davies-Bouldin Index is not less than 1.5")
        self.assertGreater(ch_index, expected_ch_index, msg="Calinski-Harabasz Index is not greater than 70")

if __name__ == "__main__":
    unittest.main()
