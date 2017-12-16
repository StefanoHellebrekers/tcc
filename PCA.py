import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')


def run_PCA(entry_data) :

  df = pd.read_csv(entry_data + ".csv")
  df.fillna(value = 0, inplace = True)
  df_temp = pd.DataFrame()
  df_temp = df

  for column in df_temp.columns :
    if column.startswith("user_id") | column.startswith("activity_country") | column.startswith("first_app_version") | column.startswith("last_app_version") | column.startswith("activity_date") | column.startswith("last_activity_date") | column.startswith("Unnamed: 0") :
        df_temp = df_temp.drop(column, axis = 1)

  pca = PCA()
  pca.fit(df_temp)
  explained_variance = pca.explained_variance_
  components = pca.components_
  components_df = pd.DataFrame(components).T
  values_df = df_temp
  transformed_df = pd.DataFrame(np.matmul(values_df, components_df))
  for column in transformed_df.columns :
      transformed_df.rename(columns={column: 'PC_' + str(column+1)}, inplace=True)

  transformed_df['user_id'] = df['user_id']
  transformed_df['activity_date'] = df['activity_date']
  transformed_df['last_activity_date'] = df['last_activity_date']

  explained_variance_ratio = pca.explained_variance_ratio_

  cumulative_explained_variance_ratio = np.cumsum(np.round(explained_variance_ratio, decimals=4)*100)
  cumulative_explained_variance_ratio_df = pd.DataFrame(cumulative_explained_variance_ratio)

  transformed_df.to_csv('pca_transformed_' + entry_data + ".csv")
  cumulative_explained_variance_ratio_df.to_csv('cumulative_explained_variance_ratio_' + entry_data + '.csv')
    
  return transformed_df, cumulative_explained_variance_ratio_df, components_df