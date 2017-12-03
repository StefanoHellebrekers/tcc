#!/usr/bin/python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None

"""
Derived From:
http://jpktd.blogspot.com/2012/06/non-linear-dependence-measures-distance.html
"""
def dist(x, y):
  #1d only
  # Calculate the absolute value element-wise
  # Returns: An ndarray containing the absolute value of each element in `x`
  return np.abs(x[:, None] - y)

def d_n(x):
  d = dist(x, x)
  dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean() 
  return dn

def dcov_all(x, y):
  # Coerce type to numpy array if not already of that type.
  try: x.shape
  except AttributeError: x = np.array(x)
  try: y.shape
  except AttributeError: y = np.array(y)
  
  dnx = d_n(x)
  dny = d_n(y)
    
  denom = np.product(dnx.shape)
  dc = np.sqrt((dnx * dny).sum() / denom)
  dvx = np.sqrt((dnx**2).sum() / denom)
  dvy = np.sqrt((dny**2).sum() / denom)
  dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
  return dc, dr, dvx, dvy

def distance_correlation(x,y):
  return dcov_all(x,y)[1]


def get_df_from_query(query, project="parabolic-water-186711"):
    print("Doing query")  
    df = pd.io.gbq.read_gbq(query, project)
    return df


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
    
  return transformed_df, cumulative_explained_variance_ratio_df, components_df, values_df


# Método generalizado de random forest. 
# Parâmetros : 
  # df - dataframe a ser utilizado; 
  # balance_database - booleano que define se é necessário balancear a base de treino (útil para casos extremos como, por exemplo, prever payers); default = True
  # split_ratio - taxa de split entre base de treino e base de teste; default = 75%
  # n_jobs - número de jobs a rodar em paralelo; default = -1, que significa o número total de  cores
  # n_estimators - número de árvores do modelo; o default do método é 10, aqui usam-se 1000 como default
# Resposta :
  # clf - modelo classificador, para ser aplicado em bases futuras caso obtenha alta performance
  # features_importance_df - dataframe de importância das features na decisão/predição
  # train - dataframe de treino, com resultados
  # test - dataframe de teste, com resultados
  # df_untrained - dataframe que não foi considerado como treino devido balanceamento, com resultados
  # df_full - dataframe completo, com resultados
  # training_score - accuracy do modelo em relação à base de treino
  # test_score - accuracy do modelo em relação à base de teste
  # untrained_score - accuracy do modelo em relação à base não treinada
def run_random_forest(df, column_to_predict, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000):
  print("Random Forest")
  # Dataframe temporário, sob o qual serão feitas todas modificações (para não alterar o df original)
  df_temp = df
  # Balancear a base, caso balance_database seja True
  if balance_database == True :
    df_train = pd.DataFrame()
    df_untrained = pd.DataFrame()
    minimum_occurence = df_temp[column_to_predict].value_counts().min()

    for key in df_temp[column_to_predict].value_counts().keys() :
      df_temp2 = df_temp[df_temp[column_to_predict] == key]
      occurences = df_temp2.count()[0]
      df_temp2['may_train'] = np.random.uniform(0, 1, len(df_temp2)) <= 1-(occurences - minimum_occurence)/occurences
      df_train = df_train.append(df_temp2[df_temp2['may_train'] == True])
      df_untrained = df_untrained.append(df_temp2[df_temp2['may_train'] == False])
    df_temp = df_train

  # Dividir o dataframe em teste e treino, de acordo com o split_ratio
  df_temp['is_train'] = np.random.uniform(0, 1, len(df_temp)) <= split_ratio
  train, test = df_temp[df_temp['is_train']==True], df_temp[df_temp['is_train']==False]

  train.reset_index(inplace=True)
  test.reset_index(inplace=True)
  df_untrained.reset_index(inplace=True)


  # Definir as features (variáveis preditoras)
  features = df_temp.columns.difference([column_to_predict]).difference(['is_train']).difference(['user_id']).difference(['may_train']).difference(['activity_country']).difference(['last_activity_date']).difference(['activity_date']).difference(['first_app_version']).difference(['last_app_version']).difference(['activity_country'])


  # Construir o classificador
  clf = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators)

  # Construir o modelo em cima do classificador, com base na predição do valor de column_to_predict de acordo com os valores das features
  clf.fit(train[features], train[column_to_predict])

  # Predizer os valores para a base de treino
  # "The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates.""
  training_prediction = clf.predict(train[features])

  # Constrói uma tabela comparativa entre o valor predito e o valor atual
  training_results = pd.crosstab(train[column_to_predict], training_prediction, rownames=['Actual State'], colnames=['Predicted State'])

  # Calcula a acurácia média do modelo para a base de treino 
  training_score = clf.score(train[features], train[column_to_predict], )

  # Calcula probabilidade de predição para cada entrada
  training_predicted_proba = clf.predict_proba(train[features])

  # Marca o tipo de dado como treino
  train['data_type'] = 'train'
  # Adiciona as probabilidades de cada entrada ser de cada valor
  df_training_predicted_proba = pd.DataFrame(training_predicted_proba)

  for column in df_training_predicted_proba.columns :
    train['predicted_probabilitie_'+str(column)] = df_training_predicted_proba[column]
    
  # Adiciona o valor predito para cada entrada
  train['predicted_value'] = training_prediction



  # Repetir fluxo para a base de teste
  test_prediction = clf.predict(test[features])
  test_results = pd.crosstab(test[column_to_predict], test_prediction, rownames=['Actual State'], colnames=['Predicted State'])
  test_score = clf.score(test[features], test[column_to_predict], )
  test_predicted_proba = clf.predict_proba(test[features])
  
  test['data_type'] = 'test'
  df_test_predicted_proba = pd.DataFrame(test_predicted_proba)
  for column in df_test_predicted_proba.columns :
    test['predicted_probabilitie_'+str(column)] = df_test_predicted_proba[column]
  test['predicted_value'] = test_prediction


  # Repetir fluxo para a base não treinada
  untrained_predictions = clf.predict(df_untrained[features])
  untrained_results = pd.crosstab(df_untrained[column_to_predict], untrained_predictions, rownames=['Actual State'], colnames=['Predicted State'])
  untrained_score = clf.score(df_untrained[features], df_untrained[column_to_predict], )
  untrained_predicted_proba = clf.predict_proba(df_untrained[features])
  df_untrained['data_type'] = 'untrained'
  df_untrained_predicted_proba = pd.DataFrame(untrained_predicted_proba)
  for column in df_untrained_predicted_proba.columns :
    df_untrained['predicted_probabilitie_'+str(column)] = df_untrained_predicted_proba[column]
  df_untrained['predicted_value'] = untrained_predictions

  # Definir importância das features
  features_importance_df = pd.DataFrame(features)
  features_importance_df['feature_importance'] = clf.feature_importances_
  features_importance_df.rename(columns={0:'feature'}, inplace=True)

  # Salvar todos os dataframes, com resultados, em um csv
  df_full = test.append(train).append(df_untrained)
  #df_full.to_csv('random_forest_results_' + entry_data + '.csv')

  return clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results


# Método para prever retenção
def predict_retention (df, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000) :
  df.fillna(0, inplace = True)
  df.loc[df.last_activity_date > df.activity_date, 'retained?'] = 1
  df.loc[df.last_activity_date <= df.activity_date, 'retained?'] = 0

  clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results = run_random_forest(df, 'retained?', balance_database, split_ratio, n_jobs, n_estimators)
  return clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results


def collect_random_forest_retention_with_pca_results(entry_data, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000):
  run_PCA(entry_data)
  transformed_df = pd.read_csv('pca_transformed_' + entry_data + ".csv")
  transformed_df = transformed_df.drop("Unnamed: 0", axis = 1)

  pca_df = pd.DataFrame()
  pca_df['user_id'] = transformed_df['user_id']
  pca_df['activity_date'] = transformed_df['activity_date']
  pca_df['last_activity_date'] = transformed_df['last_activity_date']

  results_df = pd.DataFrame()

  i = 1
  while i <= (transformed_df.columns.size-3) :
    print(entry_data + "- Doing component " + str(i))
    column_name = 'PC_' + str(i)
    print(column_name)
    pca_df[column_name] = transformed_df[column_name]
    clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results = predict_retention(pca_df, balance_database, split_ratio, n_jobs, n_estimators)
    
    results_df['Components'] = i
    results_df['test_score'] = test_score
    results_df['untrained_score'] = untrained_score
    i += 1

  results_df.to_csv('pca_results_' + entry_data + ".csv")

def collect_random_forest_retention_with_distance_correlation_results(entry_data) :
  df = pd.read_csv(entry_data + ".csv")
  df.loc[df.last_activity_date > df.activity_date, 'retained?'] = 1
  df.loc[df.last_activity_date <= df.activity_date, 'retained?'] = 0
  
  results_df = pd.DataFrame()
  correlation_df = pd.DataFrame()

  i = 0.0
  while i <= 1 :
    print(entry_data + "- Doing correlation " + str(i))
    correlation_df = df_with_selected_correlation_only(df, "retained?", i)
    print("Got correlated df")
    clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results = predict_retention(correlation_df, balance_database, split_ratio, n_jobs, n_estimators)
    print(test_score)
    results_df['Correlation'] = i
    results_df['test_score'] = test_score
    results_df['untrained_score'] = untrained_score
    i += 0.1

  results_df.to_csv('distance_correlation_results_' + entry_data + ".csv")

# Método para prever métrica em um determinado dia ativo de cada usuário, para todo usuário que teve aquele dia ativo. Já remove usuários com menos dias ativos do que o dia a ser analisado
# Parâmetros:
  # df - dataframe
  # day_to_predict - dia (d0, ... , dn) a ser predito
  # csv_prefix - prefixo dos csvs salvos
  # minimum_app_version - app version mínima
  # minimum_days_since_creation -  mínimo de dias desde criação 
# Regra de uso : o dataframe deve conter os campos user_id, first_app_version, lifetime_active_days, days_since_creation, e a coluna a ser predita.
  # Além disso, deve conter campos com métricas diárias de todos os dias até o dia a ser analisado 
  # (Ex.: daily_session_events_d0, daily_session_events_d1, daily_session_events_d2, ...)
  # Estes campos, junto com first_app_version, serão usados como preditores de retenção dos usuários

def predict_daily_metric_with_random_forest(df, column_to_predict, day_to_predict = 0, csv_prefix = "", minimum_app_version = "1.0", minimum_days_since_creation = 10, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000) : 
    # Removing unwanted data
    df_temp = df[(df['first_app_version'] >= minimum_app_version) & (df['lifetime_active_days'] >= day_to_predict+1) & (df['days_since_creation'] >= minimum_days_since_creation)]
    
    # Select all daily metrics of interest
    day = 0
    metrics = []
    while day <= day_to_predict :
        for column in df_temp.columns :
            if column.endswith("_d" + str(day)) and column != column_to_predict :
                metrics.append(column) 
        day += 1
    # Appending all other metrics of interest
    for metric in ['user_id', column_to_predict] :
        metrics.append(metric)
    df_temp = df_temp[metrics]
    clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results = run_random_forest(df_temp, column_to_predict, csv_prefix)
    test.to_csv(csv_prefix + "predict_" + column_to_predict + "_df_test_d" + str(day_to_predict) + ".csv")
    train.to_csv(csv_prefix + "predict_" + column_to_predict + "_df_train_d" + str(day_to_predict) + ".csv")
    df_untrained.to_csv(csv_prefix + "predict_" + column_to_predict + "_df_untrained_d" + str(day_to_predict) + ".csv")
    df_full.to_csv(csv_prefix + "predict_" + column_to_predict + "_df_full_d" + str(day_to_predict) + ".csv")    
    return clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results


# Método que itera predict_daily_metric_with_random_forest para cada dia dentre os dias escolhidos (de 0 à days_to_predict)
# IMPORTANTE!!! Caso seja desejado prever uma métrica para cada um dos dias ativos, column_to_predict deve terminar em "_d"!
  # Exemplo: Prever retenção dia a dia -> column_to_predict = "retained_d" 
def predict_metric_for_user_period_with_random_forest(df, column_to_predict, days_to_predict, csv_prefix = "", minimum_app_version = "1.0", minimum_days_since_creation = 10, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000) :
    day = 0
    while day <= days_to_predict :
      if column_to_predict.endswith("_d") :
        predict_daily_metric_with_random_forest(df, column_to_predict + str(day), day, csv_prefix, minimum_app_version, minimum_days_since_creation, balance_database, split_ratio, n_jobs, n_estimators)
      else :
        predict_daily_metric_with_random_forest(df, column_to_predict, day, csv_prefix, minimum_app_version, minimum_days_since_creation, balance_database, split_ratio, n_jobs, n_estimators)        
      day += 1

# DEPRACATED
# Roda método de random forest entre cada coluna e a coluna a ser predita (AVISO : não performa tão bem quanto rodar com todas as colunas)
def run_random_forest_between_columns(df, column_to_predict, split_ratio = .75, n_jobs=-1, n_estimators=1000):

    for column in df : 
        if column != column_to_predict and column != 'is_train':
            df_2 = df[[column, column_to_predict]]
            # Spliting the data frame
            df_2['is_train'] = np.random.uniform(0, 1, len(df_2)) <= split_ratio
            train, test = df_2[df_2['is_train']==True], df_2[df_2['is_train']==False]

            # Defining features (all but column_to_predict)
            features = train.columns.difference([column_to_predict]).difference(['is_train']).difference(['user_id'])
            # Construct the classifier
            if features.size > 0 :
                clf = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators)

                # Build a forest of trees from the training set (train[features], train[column_to_predict])
                clf.fit(train[features], train[column_to_predict])

                # Predict class for test[features]. 
                #The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates.
                preds = clf.predict(train[features])
                # Computes a single table to compare actual state to predicted state
                results = pd.crosstab(test[column_to_predict], preds, rownames=['Actual State'], colnames=['Predicted State'])

                # Computes the mean accuracy on the given test data and labels 
                score = clf.score(train[features], train[column_to_predict], )
    return clf, score

# Retorna um df apenas com colunas de correlação de distância maior ou igual à requerida
def df_with_selected_correlation_only(df, column_to_predict, required_correlation) :
    df_with_selected_correlation = df
    for column in df_with_selected_correlation :
        if column != column_to_predict and column != 'user_id' and column != 'is_train' and column != 'may_train' and column != 'activity_country' and column != 'last_activity_date' and column != 'activity_date' and column != 'first_app_version' and column != 'last_app_version':
            if distance_correlation(df_with_selected_correlation[column].tolist(), df_with_selected_correlation[column_to_predict].tolist()) < required_correlation:
                df_with_selected_correlation.drop(column, axis=1, inplace=True)
    return df_with_selected_correlation


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

def collect_random_forest_with_pca_results_full() :
  collect_random_forest_retention_with_pca_results("bear_evolution_d0_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d1_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d2_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d3_joined")
  collect_random_forest_retention_with_pca_results("bear_evolution_d4_joined")
def  collect_random_forest_retention_with_pca_results() :
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



if __name__ == '__main__':
  
  # Preparing the data takes a very long time
  #prepare_all_data()
  join_all_data()  
  print("yo")  
  print("yo")  
  #prepare_entry_date("bear_evolution_d4")