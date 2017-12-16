#!/usr/bin/python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')

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
def run_random_forest(df, data_name, column_to_predict, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000):
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
  features = df_temp.columns
  for column in features :
    if (column == column_to_predict) | column.startswith("user_id") | column.startswith("may_train") | column.startswith("activity_country") | column.startswith("is_train") | column.startswith("first_app_version") | column.startswith("last_app_version") | column.startswith("activity_date") | column.startswith("last_activity_date") | column.startswith("Unnamed: 0") :
      features = features.drop(column)

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
  df_full.to_csv('random_forest_results_' + data_name + '.csv')

  return clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results


# Método para prever retenção
def predict_retention (df, data_name, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000) :
  df.fillna(0, inplace = True)
  df.loc[df.last_activity_date > df.activity_date, 'retained?'] = 1
  df.loc[df.last_activity_date <= df.activity_date, 'retained?'] = 0

  clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results = run_random_forest(df, data_name,'retained?', balance_database, split_ratio, n_jobs, n_estimators)
  return clf, features_importance_df, train, test, df_untrained, df_full, training_score, test_score, untrained_score, training_results, test_results, untrained_results

    
def collect_random_forest_retention_with_pca_results(entry_data, balance_database = True, split_ratio = .75, n_jobs=-1, n_estimators=1000):
  print("Random forest with pca for " + entry_data)
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


# Retorna um df apenas com colunas de correlação de distância maior ou igual à requerida
def df_with_selected_correlation_only(df, column_to_predict, required_correlation) :
    df_with_selected_correlation = df.fillna(value = 0, inplace = True)
    for column in df_with_selected_correlation :
        if column != column_to_predict and column != 'user_id' and column != 'is_train' and column != 'may_train' and column != 'activity_country' and column != 'last_activity_date' and column != 'activity_date' and column != 'first_app_version' and column != 'last_app_version':
            if distance_correlation(df_with_selected_correlation[column].values, df_with_selected_correlation[column_to_predict].values) < required_correlation:
                df_with_selected_correlation.drop(column, axis=1, inplace=True)
    return df_with_selected_correlation

def save_result_columns_from_random_forest(entry_data) :
    df = pd.read_csv("random_forest_results_" + entry_data + ".csv")
    df_new = pd.DataFrame()
    df_new['user_id'] = df['user_id']
    df_new['data_type'] = df['data_type']
    df_new['predicted_probabilitie_0'] = df['predicted_probabilitie_0']
    df_new['predicted_probabilitie_1'] = df['predicted_probabilitie_1']
    df_new['predicted_value'] = df['predicted_value']
    df_new['retained?'] = df['retained?']
    df_new.to_csv("random_forest_result_columns_" + entry_data + ".csv")