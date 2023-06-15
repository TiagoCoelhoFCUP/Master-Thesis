import numpy as np
import pandas as pd
import pickle

seed = 15
np.random.seed(seed)

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('df_en2.csv',index_col=0, encoding = "UTF-8")
df['created_at'] = pd.to_datetime(df.created_at)
df['topic'] = df.topic.astype('category')
df = df.reset_index()

def get_weekday(row):
  return row['created_at'].weekday()

def get_hour(row):
  return row['created_at'].hour

def get_month(row):
  return row['created_at'].month

def get_retweeted_status(row):
  if str(row['retweeted_status.text']) == 'nan':
    return 0
  else:
    return 1

df['institution'] = len(df) - pd.Categorical(df["user.name"], ordered=True).codes
df['weekday'] = df.apply(get_weekday, axis=1)
df['hour'] = df.apply(get_hour, axis=1)
df['month'] = df.apply(get_month, axis=1)
df['is_retweet_status'] = df.apply(get_retweeted_status, axis=1)
df['is_quote_status'] = df['is_quote_status'] * 1

#One hot encode categorical variables
temp = pd.get_dummies(df.weekday)
temp.columns=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.topic)
temp.columns=['dominant_topic_0','dominant_topic_1','dominant_topic_2','dominant_topic_3','dominant_topic_4','dominant_topic_5']
df = pd.concat([df,temp], axis=1)

def get_quarter(row):
  if row['month'] < 4:
    return "1st Quarter"
  if row['month'] < 7:
    return "2nd Quarter"
  if row['month'] < 10:
    return "3rd Quarter"
  else:
    return "4th Quarter"

def get_time_of_day(row):
  if row['hour'] >= 8 and row['hour'] <= 15:
    return "Day"
  if row['hour'] >= 16 and row['hour'] <= 23:
    return "Afternoon/Evening"
  else:
    return "Night"

df['time_of_day'] = df.apply(get_time_of_day, axis=1)
df['quarter'] = df.apply(get_quarter, axis=1)

temp = pd.get_dummies(df.time_of_day)
temp.columns=['Day','Afternoon/Evening','Night']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.quarter)
temp.columns=['1st Quarter','2nd Quarter','3rd Quarter','4th Quarter']
df = pd.concat([df,temp], axis=1)

cols = ['retweet_count'] #outlier columns

lista = []
for string in df['user.name'].unique():
  df_temp = df[df['user.name'] == string ]
  Q1 = df_temp[cols].quantile(0.25)
  Q3 = df_temp[cols].quantile(0.75)
  IQR = Q3 - Q1
  df_temp = df_temp[~((df_temp[cols] < (Q1 - 1.5 * IQR)) |(df_temp[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
  lista.append(df_temp)

df = pd.concat(lista)

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

x = df[['is_quote_status','is_retweet_status','topic','topic_perc','topic_0','topic_1','topic_2','topic_3','topic_4','topic_5','polarity','weekday','hour','month','user.followers_count']] #independent columns
y = df['retweet_count']    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_regression, k='all')
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

import plotly.express as px

featureScores.drop(featureScores.loc[featureScores['Score'] < 0.041].index, inplace=True)
featureScores = featureScores.sort_values(["Score"], ascending=False)
fig = px.bar(featureScores, x='Specs', y='Score',title="Feature Importance (Experiment 2)",labels={'Specs':'Features','Score':'Score'})
fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1000)
fig.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


x = df[['topic_0','topic_1','topic_2','topic_3','topic_4','topic_5','topic_perc','is_retweet_status','polarity','user.followers_count']].to_numpy()
#x = df[['is_quote_status','is_retweet_status','topic_perc','topic_0','topic_1','topic_2','topic_3', 'topic_4']].to_numpy()
y = df['retweet_count'].to_numpy()

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

### This is the train and test to be used by all models. Cross Validation in the train. Then test with best models ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 
model_list = []

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
#scaling = StandardScaler().fit(x_train)
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)

##Skiped the hyperparameter optimization phase for time sake. The following models are optimized according to multiple grid_searches.

"""Multiple Linear Regression"""
from sklearn.linear_model import LinearRegression
linear = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

"""Ridge Regression"""
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.4, copy_X=True, fit_intercept=True, normalize=False, solver='sparse_cg')

"""Lasso Regression"""
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.0, copy_X=True, fit_intercept=True, normalize=True, selection='cyclic', warm_start=True)

"""Decision Tree"""
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(ccp_alpha=0.225, criterion='friedman_mse', max_depth=15, max_features='auto', min_impurity_decrease=0.1, min_samples_split=6, splitter='random')

"""Decision Tree Bagging"""
from sklearn.ensemble import BaggingRegressor
bagging = BaggingRegressor(bootstrap=True, bootstrap_features=False, max_features=8, max_samples=4994, n_estimators=93, warm_start=False)

"""Adaptive Boosting"""
from sklearn.ensemble import AdaBoostRegressor
boosting = AdaBoostRegressor(learning_rate=0.3535353535353536, loss='linear', n_estimators=5)

"""KNN"""
from sklearn.neighbors import KNeighborsRegressor

x = np.nan_to_num(x)
knn = KNeighborsRegressor(algorithm='ball_tree', leaf_size=1, metric='manhattan', n_neighbors=45, weights='distance')

"""SVM"""
from sklearn.svm import SVR
svm = SVR(C=100, degree=1, kernel='rbf', cache_size=7000)

"""Random Forest"""
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(criterion='mse', max_depth=10, max_features='auto', min_samples_split=10, n_estimators=100)

"""MLP"""
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(activation='tanh', early_stopping=False, hidden_layer_sizes=(40, 100, 100), learning_rate='adaptive')


"""Model Comparisson"""
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
import math


model_list = []
model_list.append(("Linear Regression",linear))
model_list.append(("Ridge Regression",ridge))
model_list.append(("Lasso Regression",lasso))
model_list.append(("Decision Tree",dt))
model_list.append(("Decision Tree Bagging",bagging))
model_list.append(("Decision Tree Boosting",boosting))
model_list.append(("KNN",knn))
model_list.append(("SVM",svm))
model_list.append(("Random Forest",rf))
model_list.append(("MLP",mlp))

results_df = pd.DataFrame()

for pair in model_list:
  model = pair[1].fit(x_train, y_train)
  preds = model.predict(x_test)
  R2 = r2_score(y_test, preds)
  MSE = mean_squared_error(y_test, preds)
  RMSE = math.sqrt(mean_squared_error(y_test, preds))
  MAE = mean_absolute_error(y_test, preds)
  
  results_df = results_df.append(pd.Series([pair[0], round(MSE,4), round(RMSE,4), round(MAE,4),round(R2,4)]), ignore_index=True)

results_df.columns = ['Model','MSE', 'RMSE', 'MAE', 'R2']
results_df

import plotly.graph_objects as go
df_temp = results_df
fig = go.Figure(data=[go.Table(header=dict(values=list(df_temp.columns),fill_color='paleturquoise',align='left'),cells=dict(values=df_temp.transpose().values.tolist(),fill_color='lavender',align='left'))])
fig.show()