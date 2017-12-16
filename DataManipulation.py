import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')

def prepare_entry_data(entry_data):
  entry_df = pd.read_csv(entry_data + ".csv")
  entry_df.fillna(0, inplace=True)
  features_df = pd.read_csv((entry_data + "_features.csv"))
  exit_df = pd.DataFrame()
  filtered_df = pd.DataFrame()

  i = 1
  total = entry_df['user_id'].unique().size
  for user in entry_df['user_id'].unique() :
    user_df = entry_df[entry_df['user_id'] == user]
    print("Preparing user ", i, " of ", total)
    #DF Temporario
    temp_df = pd.DataFrame()

    # Setar caracteristicar de usuario
    temp_df.set_value(0, 'user_id', user)

    temp_df.set_value(0, 'activity_country', user_df['activity_country'].iloc[0])
    temp_df.set_value(0, 'first_app_version', user_df['first_app_version'].iloc[0])
    temp_df.set_value(0, 'last_app_version', user_df['last_app_version'].iloc[0])
    temp_df.set_value(0, 'last_activity_date', user_df['last_activity_date'].iloc[0])
    temp_df.set_value(0, 'activity_date', user_df['activity_date'].iloc[0])

    for index, row in features_df.iterrows() :

      if row['param_type'] == "String" :

        feature_name = row['event_name'] + "-" + row['param_key'] + "-" + row['string_value']
        filtered_df = user_df[(user_df['event_name'] == row['event_name']) & (user_df['param_key'] == row['param_key']) & (user_df['string_value'] == row['string_value'])]

        occurences = len(filtered_df.axes[0])
        temp_df[feature_name] = occurences
      else :
        if row['param_key'] == 'engagement_time_msec' :

          feature_name = row['event_name'] + "-engagement_time_sec"
          filtered_df = user_df[(user_df['event_name'] == row['event_name']) & (user_df['param_key'] == row['param_key'])]

          temp_df[feature_name + "_count"] = len(filtered_df.axes[0])
          temp_df[feature_name + "_sum"] = (filtered_df['double_value']/1000000).sum()
          temp_df[feature_name + "_mean"] = (filtered_df['double_value']/1000000).mean()
          temp_df[feature_name + "_std_deviation"] = (filtered_df['double_value']/1000000).std()
          temp_df[feature_name + "_max_value"] = (filtered_df['double_value']/1000000).max()
          temp_df[feature_name + "_min_value"] = (filtered_df['double_value']/1000000).min()
        else :
          feature_name = row['event_name'] + "-" + row['param_key']
          filtered_df = user_df[(user_df['event_name'] == row['event_name']) & (user_df['param_key'] == row['param_key'])]

          temp_df[feature_name + "_count"] = len(filtered_df.axes[0])
          temp_df[feature_name + "_sum"] = filtered_df['double_value'].sum()
          temp_df[feature_name + "_mean"] = filtered_df['double_value'].mean()
          temp_df[feature_name + "_std_deviation"] = filtered_df['double_value'].std()
          temp_df[feature_name + "_max_value"] = filtered_df['double_value'].max()
          temp_df[feature_name + "_min_value"] = filtered_df['double_value'].min()

    exit_df = exit_df.append(temp_df)
    i += 1

  exit_df.to_csv(entry_data + '_prepared.csv')


  return exit_df


# Example : table_name = "bear_evolution"
def join_prepared_data(table_name) :

  d0 = pd.read_csv(table_name + '_d0_prepared.csv')
  d0 = d0.drop("Unnamed: 0", axis = 1)
  for column in d0.columns :
    if column != "user_id" and column != "activity_date" and column != "last_activity_date":
        d0.rename(columns={column: column + '_d0'}, inplace = True)
  d0.to_csv(table_name + '_d0_joined.csv')

  d1 = pd.read_csv(table_name + '_d1_prepared.csv')
  d1 = d1.drop("Unnamed: 0", axis = 1)
  for column in d1.columns :
    if column != "user_id" and column != "activity_date" and column != "last_activity_date":
        d1.rename(columns={column: column + '_d1'}, inplace = True)
  for column in d0.columns :
    if column.startswith("activity_country") | column.startswith("first_app_version") | column.startswith("last_app_version") | column.startswith("activity_date") | column.startswith("last_activity_date") | column.startswith("Unnamed: 0") :
        d0 = d0.drop(column, axis = 1)        
  d1 = pd.merge(d0, d1, on='user_id', how='inner')
  d1.to_csv(table_name + '_d1_joined.csv')

  d2 = pd.read_csv(table_name + '_d2_prepared.csv')
  d2 = d2.drop("Unnamed: 0", axis = 1)
  for column in d2.columns :
    if column != "user_id" and column != "activity_date" and column != "last_activity_date":
        d2.rename(columns={column: column + '_d2'}, inplace = True)
  for column in d1.columns :
    if column.startswith("activity_country") | column.startswith("first_app_version") | column.startswith("last_app_version") | column.startswith("activity_date") | column.startswith("last_activity_date") | column.startswith("Unnamed: 0") :
        d1 = d1.drop(column, axis = 1)
  d2 = pd.merge(d1, d2, on='user_id', how='inner')
  d2.to_csv(table_name + '_d2_joined.csv')

  d3 = pd.read_csv(table_name + '_d3_prepared.csv')
  d3 = d3.drop("Unnamed: 0", axis = 1)
  for column in d3.columns :
    if column != "user_id" and column != "activity_date" and column != "last_activity_date":
        d3.rename(columns={column: column + '_d3'}, inplace = True)
  for column in d2.columns :
    if column.startswith("activity_country") | column.startswith("first_app_version") | column.startswith("last_app_version") | column.startswith("activity_date") | column.startswith("last_activity_date") | column.startswith("Unnamed: 0") :
        d2 = d2.drop(column, axis = 1)
  d3 = pd.merge(d2, d3, on='user_id', how='inner')
  d3.to_csv(table_name + '_d3_joined.csv')

  d4 = pd.read_csv(table_name + '_d4_prepared.csv')
  d4 = d4.drop("Unnamed: 0", axis = 1)
  for column in d4.columns :
    if column != "user_id" and column != "activity_date" and column != "last_activity_date":
        d4.rename(columns={column: column + '_d4'}, inplace = True)
  for column in d3.columns :
    if column.startswith("activity_country") | column.startswith("first_app_version") | column.startswith("last_app_version") | column.startswith("activity_date") | column.startswith("last_activity_date") | column.startswith("Unnamed: 0") :
        d3 = d3.drop(column, axis = 1)
  d4 = pd.merge(d3, d4, on='user_id', how='inner')
  d4.to_csv(table_name + '_d4_joined.csv')