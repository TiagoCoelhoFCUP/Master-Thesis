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

df = pd.read_csv('df_en.csv',index_col=0, encoding = "UTF-8")
df['created_at'] = pd.to_datetime(df.created_at)
df['topic'] = df.topic.astype('category')
#df['diversity'] = df.diversity.astype('category')
df = df.reset_index()
df = df.sort_values('created_at')

df_new = df[['created_at','polarity','diversity','topic','topic_perc','user.name','user.followers_count','retweet_count','favorite_count']]
temp = df_new.set_index('created_at').groupby([pd.Grouper(freq='D'),'user.name'])['topic'].apply(list).reset_index()

def compute_mode(numbers):
    lista = []
    counts = {}
    maxcount = 0
    for number in numbers:
        if number not in counts:
            counts[number] = 0
        counts[number] += 1
        if counts[number] > maxcount:
            maxcount = counts[number]

    for number, count in counts.items():
        if count == maxcount:
            lista.append(str(number))
    lista.sort()
    return lista


def assign_mode(row):
  lista = compute_mode(row['topic'])
  if len(lista) != 1:
    return -1
  else:
    return lista[0]

temp['topic'] = temp.apply(assign_mode, axis=1)
temp2 = df_new.set_index('created_at').groupby([pd.Grouper(freq='D'),'user.name'])['topic_perc','diversity','favorite_count','retweet_count','polarity'].mean().reset_index()
df_new = pd.merge(temp, temp2, on=['created_at','user.name'], how='outer')
df_new = df_new[df_new.topic != -1]
df_new = df_new.reset_index(drop=True)

import plotly.graph_objects as go
df_temp = df_new
fig = go.Figure(data=[go.Table(header=dict(values=list(df_temp.columns),fill_color='paleturquoise',align='left'),cells=dict(values=df_temp.transpose().values.tolist(),fill_color='lavender',align='left'))])
fig.show()

avg = []
for name in df_new['user.name'].unique():
  df_temp = df_new[df_new['user.name'] == name].reset_index(drop=True)
  dic = {}
  for i in range(len(df_temp)-2):
    topic1 = df_temp.at[i,'topic']
    topic2 = df_temp.at[i+1,'topic']
    topic3 = df_temp.at[i+2,'topic']
    lista = [topic1, topic2, topic3]
    pattern = " ".join(lista)
    if pattern not in dic:
      dic[pattern] = 1
    else:
      dic[pattern] += 1
  avg.append(len(dic))

df_new['month'] = pd.DatetimeIndex(df_new['created_at']).month
df_new['year'] = pd.DatetimeIndex(df_new['created_at']).year
df_new['weekday'] = df_new.apply(lambda row: row['created_at'].weekday(), axis=1)

lista = []
for string in sorted(df_new['user.name'].unique()):
  temp = df_new[df_new['user.name'] == string].reset_index()
  temp['index'] = temp.index
  new_df = pd.DataFrame()
  i = 0
  while i <= len(temp)-8:
    seven_tweets = temp[(temp['index'] >= i) & (temp['index'] <= i+7)]
    day1 = seven_tweets['topic'].to_numpy()[0]
    day2 = seven_tweets['topic'].to_numpy()[1]
    day3 = seven_tweets['topic'].to_numpy()[2]
    day4 = seven_tweets['topic'].to_numpy()[3]
    day5 = seven_tweets['topic'].to_numpy()[4]
    day6 = seven_tweets['topic'].to_numpy()[5]
    day7 = seven_tweets['topic'].to_numpy()[6]
    year =  seven_tweets['year'].to_numpy()[-1]
    weekday = seven_tweets['weekday'].to_numpy()[-1]
    month = seven_tweets['month'].to_numpy()[-1]
    avg_retweets = seven_tweets['retweet_count'][:-1].mean()
    avg_favorites = seven_tweets['favorite_count'][:-1].mean()
    target_topic = seven_tweets['topic'].to_numpy()[-1]
    institution =  string
    new_df = new_df.append(pd.Series([day1,day2,day3,day4,day5,day6,day7,int(weekday),int(year),int(month),float(round(avg_retweets,2)),float(round(avg_favorites,2)),str(institution),target_topic]), ignore_index=True)
    i += 1

  new_df.columns = ['Day1','Day2','Day3','Day4','Day5','Day6','Day7','Weekday','Year','Month','Avg Retweets','Avg Favorites','Institution','Target Topic']
  new_df['Weekday'] = new_df['Weekday'].astype('category')
  new_df['Year'] = new_df['Year'].astype('category')
  new_df['Month'] = new_df['Month'].astype('category')
  new_df['Avg Retweets'] = new_df['Avg Retweets'].astype(float)
  new_df['Avg Favorites'] = new_df['Avg Favorites'].astype(float)
  new_df['Target Topic'] = new_df['Target Topic'].astype('category')
  new_df['Day1'] = new_df['Day1'].astype('category')
  new_df['Day2'] = new_df['Day2'].astype('category')
  new_df['Day3'] = new_df['Day3'].astype('category')
  new_df['Day4'] = new_df['Day4'].astype('category')
  new_df['Day5'] = new_df['Day5'].astype('category')
  new_df['Day6'] = new_df['Day6'].astype('category')
  new_df['Day7'] = new_df['Day7'].astype('category')
  new_df['Institution'] = new_df['Institution'].astype('category')
  lista.append(new_df)

df = pd.concat(lista)
df = df.reset_index(drop=True)

df["Institution"] = df['Institution'].astype('category')
df["Institution"] = df["Institution"].cat.codes
#One hot encode categorical variables
temp = pd.get_dummies(df.Institution)
temp.columns=["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UCL", "University of Oxford","UC Berkeley"]
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day1)
temp.columns=['day1_0','day1_1','day1_2','day1_3','day1_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day2)
temp.columns=['day2_0','day2_1','day2_2','day2_3','day2_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day3)
temp.columns=['day3_0','day3_1','day3_2','day3_3','day3_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day4)
temp.columns=['day4_0','day4_1','day4_2','day4_3','day4_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day5)
temp.columns=['day5_0','day5_1','day5_2','day5_3','day5_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day6)
temp.columns=['day6_0','day6_1','day6_2','day6_3','day6_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Day7)
temp.columns=['day7_0','day7_1','day7_2','day7_3','day7_4']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Weekday)
temp.columns=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Month)
temp.columns=['january','february','march','april','may','july','june','august','september','october','november','december']
df = pd.concat([df,temp], axis=1)

temp = pd.get_dummies(df.Year)
temp.columns=['2019','2020']
df = pd.concat([df,temp], axis=1)


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

x = df[['Avg Retweets','Avg Favorites',"Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UCL", "University of Oxford","UC Berkeley","day1_0",	"day1_1",	"day1_2",	"day1_3",	"day1_4",	"day2_0",	"day2_1",	"day2_2",	"day2_3",	"day2_4",	"day3_0",	"day3_1",	"day3_2", "day3_3", "day3_4", "day4_1","day4_2","day4_3","day4_4","day5_0",	"day5_1",	"day5_2",	"day5_3",	"day5_4","day6_0",	"day6_1",	"day6_2",	"day6_3",	"day6_4", "day7_0",	"day7_1",	"day7_2",	"day7_3",	"day7_4",	"monday",	"tuesday",	"wednesday",	"thursday",	"friday",	"saturday",	"sunday",	"january"	,"february"	,"march",	"april",	"may",	"july",	"june",	"august",	"september",	"october",	"november",	"december",	"2019",	"2020"]] #independent columns
y = df['Target Topic']    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_classif, k='all')
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

import plotly.express as px

featureScores.drop(featureScores.loc[featureScores['Score'] < 0.02777].index, inplace=True)
featureScores = featureScores.sort_values(["Score"], ascending=False)
fig = px.bar(featureScores, x='Specs', y='Score',title="Feature Importance",labels={'Specs':'Features','Score':'Score'})
fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1000)
fig.show()

x = df[["Avg Retweets","day3_2","day5_2","2020","day6_1","day6_2","day1_2","Avg Favorites","day7_2","tuesday","day2_2","day2_1","day7_1","UC Berkeley","2019","day3_1","Caltech","day4_1","MIT","day3_4"]].to_numpy()
y = df['Target Topic'].to_numpy()

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

### This is the train and test to be used by all models. Cross Validation in the train. Then test with best models ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 
model_list = []

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
scaling = StandardScaler().fit(x_train)
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)

"""Logistic Regression"""
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(class_weight='balanced',fit_intercept=False,max_iter=50,multi_class='ovr',solver='sag')

"""Decision Tree"""
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(ccp_alpha=0.0, criterion='gini',max_depth=30,max_features='log2', min_impurity_decrease=0.0, min_samples_leaf=10, min_samples_split=45, splitter='random')

"""DT Bagging"""
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(bootstrap=True, bootstrap_features=True, max_features=12, max_samples=71, n_estimators=79)

"""DT Boosting"""
from sklearn.ensemble import AdaBoostClassifier
boosting = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.18367346938775508, n_estimators=63)

"""Random Forest"""
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(ccp_alpha=0.0, max_depth=47, max_features='auto', min_impurity_decrease=0.0, min_samples_leaf=4, min_samples_split=30, n_estimators=87)

"""KNN"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='euclidean', n_neighbors=89, weights='uniform')

"""SVM"""
from sklearn.svm import SVC
svm = SVC(decision_function_shape='ovo', degree=1, gamma='scale', kernel='rbf', probability=True, shrinking=True, cache_size=7000)

"""MLP"""
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='identity', early_stopping=True, hidden_layer_sizes=(100, 50, 10), learning_rate='adaptive')

"""Model Comparisson"""
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


##ADD KNN, SVM e MLP
model_list = []
model_list.append(("Logistic Regression",logistic))
model_list.append(("Decision Tree",dt))
model_list.append(("Decision Tree Bagging",bagging))
model_list.append(("Decision Tree Boosting",boosting))
model_list.append(("Random Forest",rf))
model_list.append(("KNN",knn))
model_list.append(("SVM",svm))
model_list.append(("MLP",mlp))

results_df = pd.DataFrame()

for pair in model_list:
  model = pair[1].fit(x_train, y_train)
  preds = model.predict(x_test)
  acc = accuracy_score(y_test,preds)
  prec = precision_score(y_test,preds,average='macro')
  rec = recall_score(y_test,preds,average='macro')
  f1 = f1_score(y_test,preds,average='macro')
  results_df = results_df.append(pd.Series([pair[0], round(acc,4), round(prec,4), round(rec,4), round(f1,4)]), ignore_index=True)

results_df.columns = ['Model','Accuracy', 'Precision', 'Recall', 'F1']
results_df

df_temp = results_df
fig = go.Figure(data=[go.Table(header=dict(values=list(df_temp.columns),fill_color='paleturquoise',align='left'),cells=dict(values=df_temp.transpose().values.tolist(),fill_color='lavender',align='left'))])
fig.show()