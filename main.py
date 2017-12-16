#!/usr/bin/python

from DataManipulation import prepare_entry_data
from DataManipulation import join_prepared_data

from DistanceCorrelation import distance_correlation

from PCA import run_PCA

from RandomForest import run_random_forest
from RandomForest import predict_retention
from RandomForest import collect_random_forest_retention_with_pca_results
from RandomForest import collect_random_forest_retention_with_distance_correlation_results
from RandomForest import save_result_columns_from_random_forest

from HDBSCAN import cluster_with_hdbscan_after_PCA

def prepare_all_data() :
  prepare_entry_data("bear_evolution_d0")
  prepare_entry_data("bear_evolution_d1")
  prepare_entry_data("bear_evolution_d2")
  prepare_entry_data("bear_evolution_d3")
  prepare_entry_data("bear_evolution_d4")
  prepare_entry_data("dog_evolution_d0")
  prepare_entry_data("dog_evolution_d1")
  prepare_entry_data("dog_evolution_d2")
  prepare_entry_data("dog_evolution_d3")
  prepare_entry_data("dog_evolution_d4")
  prepare_entry_data("dolphin_evolution_d0")
  prepare_entry_data("dolphin_evolution_d1")
  prepare_entry_data("dolphin_evolution_d2")
  prepare_entry_data("dolphin_evolution_d3")
  prepare_entry_data("dolphin_evolution_d4")

def join_all_data() :
  join_prepared_data("bear_evolution")
  join_prepared_data("dog_evolution")
  join_prepared_data("dolphin_evolution")


def run_PCA_full() :
  run_PCA("bear_evolution_d0_joined")
  run_PCA("bear_evolution_d1_joined")
  run_PCA("bear_evolution_d2_joined")
  run_PCA("bear_evolution_d3_joined")
  run_PCA("bear_evolution_d4_joined")
  run_PCA("dog_evolution_d0_joined")
  run_PCA("dog_evolution_d1_joined")
  run_PCA("dog_evolution_d2_joined")
  run_PCA("dog_evolution_d3_joined")
  run_PCA("dog_evolution_d4_joined")
  run_PCA("dolphin_evolution_d0_joined")
  run_PCA("dolphin_evolution_d1_joined")
  run_PCA("dolphin_evolution_d2_joined")
  run_PCA("dolphin_evolution_d3_joined")
  run_PCA("dolphin_evolution_d4_joined")

def collect_random_forest_retention_with_pca_results_full() :
  collect_random_forest_retention_with_pca_results("bear_evolution_d0_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d1_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d2_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d3_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d4_joined")
  collect_random_forest_retention_with_pca_results("dog_evolution_d0_joined")
  collect_random_forest_retention_with_pca_results("dog_evolution_d1_joined")
  collect_random_forest_retention_with_pca_results("dog_evolution_d2_joined")
  collect_random_forest_retention_with_pca_results("dog_evolution_d3_joined")
  collect_random_forest_retention_with_pca_results("dog_evolution_d4_joined")
  collect_random_forest_retention_with_pca_results("dolphin_evolution_d0_joined")
  collect_random_forest_retention_with_pca_results("dolphin_evolution_d1_joined")
  collect_random_forest_retention_with_pca_results("dolphin_evolution_d2_joined")
  collect_random_forest_retention_with_pca_results("dolphin_evolution_d3_joined")
  collect_random_forest_retention_with_pca_results("dolphin_evolution_d4_joined")
    
def collect_random_forest_retention_with_distance_correlation_results_full() :
  collect_random_forest_retention_with_distance_correlation_results("bear_evolution_d0_joined")
  collect_random_forest_retention_with_distance_correlation_results("bear_evolution_d1_joined")
  collect_random_forest_retention_with_distance_correlation_results("bear_evolution_d2_joined")
  collect_random_forest_retention_with_distance_correlation_results("bear_evolution_d3_joined")
  collect_random_forest_retention_with_distance_correlation_results("bear_evolution_d4_joined")
  collect_random_forest_retention_with_distance_correlation_results("dog_evolution_d0_joined")
  collect_random_forest_retention_with_distance_correlation_results("dog_evolution_d1_joined")
  collect_random_forest_retention_with_distance_correlation_results("dog_evolution_d2_joined")
  collect_random_forest_retention_with_distance_correlation_results("dog_evolution_d3_joined")
  collect_random_forest_retention_with_distance_correlation_results("dog_evolution_d4_joined")
  collect_random_forest_retention_with_distance_correlation_results("dolphin_evolution_d0_joined")
  collect_random_forest_retention_with_distance_correlation_results("dolphin_evolution_d1_joined")
  collect_random_forest_retention_with_distance_correlation_results("dolphin_evolution_d2_joined")
  collect_random_forest_retention_with_distance_correlation_results("dolphin_evolution_d3_joined")
  collect_random_forest_retention_with_distance_correlation_results("dolphin_evolution_d4_joined")
    
def cluster_all_data() :
  cluster_with_hdbscan_after_PCA("pca_transformed_bear_evolution_d0_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_bear_evolution_d1_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_bear_evolution_d2_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_bear_evolution_d3_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_bear_evolution_d4_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dog_evolution_d0_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dog_evolution_d1_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dog_evolution_d2_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dog_evolution_d3_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dog_evolution_d4_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dolphin_evolution_d0_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dolphin_evolution_d1_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dolphin_evolution_d2_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dolphin_evolution_d3_joined")
  cluster_with_hdbscan_after_PCA("pca_transformed_dolphin_evolution_d4_joined")