#!/usr/bin/python
import pandas as pd
import hdbscan
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')    

def cluster_with_hdbscan_after_PCA(entry_data) :
  df = pd.read_csv(entry_data + ".csv")
  clustering_df = pd.DataFrame()
  clustering_df['PC_1'] = df['PC_1']
  clustering_df['PC_2'] = df['PC_2']
  cluster_size = int(len(df)/6)
  clusterer = hdbscan.HDBSCAN(cluster_size)
  clusterer.fit(clustering_df)
  labels = clusterer.labels_
  print("Number of clusters found: ", labels.max()+1)
  probabilities = clusterer.probabilities_
  clustering_df['labels'] = labels
  clustering_df['probabilities'] = probabilities
  clustering_df['user_id'] = df['user_id']
  clustering_df.to_csv(entry_data + '_hdbsca_clustering_results.csv')
  return clustering_df