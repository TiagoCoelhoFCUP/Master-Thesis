import pandas as pd
import gensim
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import adjustText

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('df_en.csv',index_col=0, encoding = "UTF-8")
df['created_at'] = pd.to_datetime(df.created_at)
df['topic'] = df.topic.astype('category')
df['diversity'] = df.diversity.astype('category')
df = df.reset_index()
df = df[df['text'].isna() == False]
df = df.reset_index(drop=True)

# Load word2vec model (trained on an enormous Google corpus)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 

# Check dimension of word vectors
model.vector_size

from nltk.tokenize import word_tokenize

# Filter out stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

texts = df['text'].tolist()
words_list = []

#Update the text in the dataframe
def update_text(row):
  tokens = word_tokenize(row['text'])
  words = [word.lower() for word in tokens if word.isalpha()]
  words = [word for word in words if not word in stop_words]
  words_filtered = [word for word in words if word in model.vocab]
  for word in words_filtered:
    words_list.append(word)
  
  return words_filtered

df['text'] = df.apply(update_text, axis=1)

# Filter the list of vectors to include only those that Word2Vec has a vector for
vector_list = [model[word] for word in words_list]

# Zip the words together with their vector representations
word_vec_zip = zip(words_list, vector_list)
 
# Cast to a dict so we can turn it into a DataFrame
word_vec_dict = dict(word_vec_zip)
vectors = pd.DataFrame.from_dict(word_vec_dict, orient='index')
vectors

#centroids

similarity = {}
centroids = ['education','employment','faculty','research','health','society']

for word in words_list:
  lista = []
  for centroid in centroids:
    lista.append(model.similarity(centroid, word))
  similarity[word] = lista

closest_words = []
topic_words = [[],[],[],[],[],[]]

for j in range(0,10):
  max_list = [-1,-1,-1,-1,-1,-1]
  closest_word = ["","","","","",""]
  for key, value in similarity.items():
    if key not in closest_words:
      for i in range(len(max_list)):
        if value[i] > max_list[i]:
          max_list[i] = value[i]
          closest_word[i] = key
  for i in range(len(closest_word)):
    closest_words.append(closest_word[i])
    topic_words[i].append(closest_word[i])

import sklearn

tweets = df['text'].tolist()
for i in range(len(tweets)):
  tweets[i] = ' '.join(tweets[i])

#instantiate CountVectorizer() 
cv=sklearn.feature_extraction.text.CountVectorizer() 
 
# this steps generates word counts for the words in your docs 
word_count_vector=cv.fit_transform(tweets)

#get idf values
tfidf_transformer=sklearn.feature_extraction.text.TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)

# count matrix 
count_vector=cv.transform(tweets) 
 
# tf-idf scores 
tf_idf_vector=tfidf_transformer.transform(count_vector)

scores_list = []
topic_perc = []

def assign_topic(row):
  #get tf-idf values for current row
  document = tf_idf_vector[row.name]
  df_temp = pd.DataFrame(document.T.todense(), index=cv.get_feature_names(), columns=['tfidf'])
  df_temp = df_temp.sort_values(by=["tfidf"],ascending=False)
  df_temp = df_temp[df_temp.tfidf != 0]
  df_temp.reset_index(level=0, inplace=True)
  values = df_temp.values.tolist()
  dic = {}
  for pair in values:
        key, value = pair[0], pair[1]
        dic[key] = value

  #calculate the score for each topic
  #mention score formula in dissertation
  scores = [0,0,0,0,0,0]
  for word in row['text']:
    if word in dic:
      for i in range(len(centroids)):
        scores[i] = scores[i] + dic[word]*similarity[word][i]
  topic = scores.index(max(scores))
  topic_perc.append(max(scores))
  scores_list.append(scores)

  #assign a topic to the current row
  #mention max assignment of topic (method)
  return topic

df['topic'] = df.apply(assign_topic, axis=1)

scores_0 = []
scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []
scores_5 = []

for lista in scores_list:
  scores_0.append(lista[0])
  scores_1.append(lista[1])
  scores_2.append(lista[2])
  scores_3.append(lista[3])
  scores_4.append(lista[4])
  scores_5.append(lista[5])

df['text'] = texts
df['topic_0'] = scores_0
df['topic_1'] = scores_1
df['topic_2'] = scores_2
df['topic_3'] = scores_3
df['topic_4'] = scores_4
df['topic_5'] = scores_5
df['topic_perc'] = topic_perc
del df['diversity']

representative_tweets = pd.DataFrame()

for i, grp in df.groupby('topic'):
    representative_tweets = pd.concat([representative_tweets, 
                                             grp.sort_values(['topic_perc'], ascending=False).head(1)], 
                                            axis=0)
representative_tweets.reset_index(drop=True, inplace=True)
representative_tweets = representative_tweets[['text','topic_perc','topic']]

#Tweets mais representativos de cada tópico
representative_tweets.style.set_properties(subset=['text'], **{'width': '1000px'})

import plotly.graph_objects as go
df_temp = representative_tweets
fig = go.Figure(data=[go.Table(header=dict(values=list(df_temp.columns),fill_color='paleturquoise',align='left'),cells=dict(values=df_temp.transpose().values.tolist(),fill_color='lavender',align='left'))])
fig.show()

df_temp = df.groupby(['topic']).size().reset_index(name='counts')

import plotly.express as px

fig = px.bar(df_temp, x='topic', y='counts', title="",labels={'topic':'Topic','counts':'Number of Tweets'})
fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1440, height=432)
fig.show()

#@title
import seaborn as sns
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

dic =  {0 : 'Education', 1 : 'Employment', 2: 'Faculty', 3: 'Research', 4: 'Health', 5: 'Society'}

fig, axes = plt.subplots(1,6,figsize=(20,6), dpi=160, sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):    
    df_dominant_topic_en_sub = df.loc[df.topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_en_sub.text]
    ax.hist(doc_lens, bins = 250, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 500), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i])
    ax.set_title(dic[i], fontdict=dict(size=20, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
#plt.xticks(np.linspace(0,1000,9))
fig.suptitle('', fontsize=30)
plt.show()

#@title

names = ["Education","Employment","Faculty","Research","Health","Society"]
topics = []

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fig = make_subplots(rows=3,cols=4,vertical_spacing=0.1,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"))

cs = 1
rs = 1
for string in sorted(df['user.name'].unique()):
  if cs > 4:
    rs += 1
    cs = 1

  lista = []
  temp = df[df['user.name'] == string ]
  for topic in range(0,6):
    temp2 = temp[temp['topic'] == topic]
    temp2 = temp2.groupby(pd.Grouper(key='created_at',freq='M'))['created_at'].agg(['count']).reset_index()
    temp2['topic'] = topic
    lista.append(temp2)

  if rs == 1 and cs == 1:
    flag = True
  else:
    flag = False
  fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name=names[0],line=dict(color=cols[0]),legendgroup="0",showlegend=flag),row=rs,col=cs)
  fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name=names[1],line=dict(color=cols[1]),legendgroup="1",showlegend=flag),row=rs,col=cs)
  fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name=names[2],line=dict(color=cols[2]),legendgroup="2",showlegend=flag),row=rs,col=cs)
  fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name=names[3],line=dict(color=cols[3]),legendgroup="3",showlegend=flag),row=rs,col=cs)
  fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name=names[4],line=dict(color=cols[4]),legendgroup="4",showlegend=flag),row=rs,col=cs)
  fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name=names[5],line=dict(color=cols[5]),legendgroup="5",showlegend=flag),row=rs,col=cs)
  topics.append((string,lista))
  cs += 1

fig.update_xaxes(
    dtick="M4",
    tickformat="%b\n%Y")

fig.update_layout(height=800,font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(visible=True, linecolor='black',showticklabels=False)
fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

for i in range(1,13):
  fig['layout']['yaxis'+str(i)].update(title='', range=[0,100], autorange=False)

#for i in range(1,13):
  #fig['layout']['xaxis'+str(i)].update(title='', range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)], autorange=False)

fig.update_annotations(font_size=20)
fig.add_vline(x="2020-03-11", line_width=3, line_dash="dash", line_color="red")
fig.show()

#@title
fig = go.Figure()

string = topics[0][0]
lista = topics[0][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[4]),legendgroup="5"))


fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[1][0]
lista = topics[1][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[2][0]
lista = topics[2][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[3][0]
lista = topics[3][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[4][0]
lista = topics[4][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[5][0]
lista = topics[5][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[6][0]
lista = topics[6][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[7][0]
lista = topics[7][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[8][0]
lista = topics[8][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[9][0]
lista = topics[9][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
fig = go.Figure()

string = topics[10][0]
lista = topics[10][1]
fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))
fig.add_trace(go.Scatter(x=lista[5]['created_at'], y=lista[5]['count'],mode='lines',name='Topic ' + str(5),line=dict(color=cols[5]),legendgroup="5"))

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y",
    range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

fig.update_layout(title=string)
fig.show()

#@title
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fig = make_subplots(rows=3,cols=4,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"))

cs = 1
rs = 1
for string in sorted(df['user.name'].unique()):
  if cs > 4:
    rs += 1
    cs = 1

  lista = []
  temp = df[df['user.name'] == string ]
  for topic in range(0,6):
    count = len(temp[temp['topic'] == topic].index)
    lista.append(count)
    
  fig.add_trace(go.Bar(x = ['Education','Employment','Faculty','Research','Health','Society'], y = lista, legendgroup="bar",showlegend=False, marker_color=cols[:6]),row=rs,col=cs)
  cs += 1

fig.update_layout(height=800,font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(visible=True, linecolor='black')
fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

for i in range(1,13):
  fig['layout']['yaxis'+str(i)].update(title='', range=[0,1000], autorange=False)

fig.update_annotations(font_size=20)
fig.show()

"""Number of Tweets"""

#@title
precov_en = df[df['created_at'] < '2020-03-11 00:00:00']

poscov_en = df[df['created_at'] >= '2020-03-11 00:00:00']

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fw3 = make_subplots(rows=1,cols=6,shared_yaxes=True,subplot_titles=("Education", "Employment", "Faculty", "Research", "Health", "Society"))

for topic in range(0,6):
  count_pre = len(precov_en[precov_en['topic'] == topic].index)
  count_pos = len(poscov_en[poscov_en['topic'] == topic].index)
  total_pre = len(precov_en.index)
  total_pos = len(poscov_en.index)
  fw3.add_trace(go.Bar(x = ['Pre', 'Post'], y = [count_pre/total_pre, count_pos/total_pos], legendgroup="bar",showlegend=False, marker_color=[cols[topic], cols[topic]]),row=1,col=topic+1)

fw3.update_annotations(font_size=25)
fw3.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fw3.update_xaxes(visible=True, linecolor='black')
fw3.update_yaxes(visible=True, linecolor='black', gridcolor='grey',tickformat=".0%")

for i in range(1,7):
  fw3['layout']['yaxis'+str(i)].update(title='',range=[0,0.3], autorange=False, )
fw3.show()

"""Number of Retweets"""

#@title
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fw4 = make_subplots(rows=1,cols=6,shared_yaxes=True,subplot_titles=("Education", "Employment", "Faculty", "Research", "Health", "Society"))

tempre_en = precov_en.groupby(['topic'])['retweet_count'].agg(['count','sum'])
tempos_en = poscov_en.groupby(['topic'])['retweet_count'].agg(['count','sum'])

for topic in range(0,6):
  fw4.add_trace(go.Bar(x = ['Pre', 'Post'], y = [tempre_en.at[topic,'sum']/tempre_en.at[topic,'count'],tempos_en.at[topic,'sum']/tempos_en.at[topic,'count']], legendgroup="bar",showlegend=False, marker_color=[cols[topic], cols[topic]]),row=1,col=topic+1)

fw4.update_annotations(font_size=25)
fw4.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fw4.update_xaxes(visible=True, linecolor='black')
fw4.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

for i in range(1,7):
  fw4['layout']['yaxis'+str(i)].update(title='',range=[0,100], autorange=False)
fw4.show()

"""Number of Favorites

"""

#@title
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fw5 = make_subplots(rows=1,cols=6,shared_yaxes=True,subplot_titles=("Education", "Employment", "Faculty", "Research", "Health", "Society"))

tempre_en = precov_en.groupby(['topic'])['favorite_count'].agg(['count','sum'])
tempos_en = poscov_en.groupby(['topic'])['favorite_count'].agg(['count','sum'])

for topic in range(0,6):
  fw5.add_trace(go.Bar(x = ['Pre', 'Post'], y = [tempre_en.at[topic,'sum']/tempre_en.at[topic,'count'],tempos_en.at[topic,'sum']/tempos_en.at[topic,'count']], legendgroup="bar",showlegend=False, marker_color=[cols[topic], cols[topic]]),row=1,col=topic+1)

fw5.update_annotations(font_size=25)
fw5.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fw5.update_xaxes(visible=True, linecolor='black')
fw5.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

for i in range(1,7):
  fw5['layout']['yaxis'+str(i)].update(title='',range=[0,100], autorange=False)
fw5.show()

"""Pre Covid"""

#@title

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fw1 = go.Figure()

names = ["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"]
topics = ["Education","Employment", "Faculty", "Research", "Health", "Society"]

for topic1 in range(0,6):
  real_lista = []
  for string in sorted(precov_en['user.name'].unique()):
    temp = precov_en[precov_en['user.name'] == string ]
    lista = []
    for topic2 in range(0,6):
      count = len(temp[temp['topic'] == topic2].index)
      lista.append(count)
    total = sum(lista)
    if total!= 0: lista[:] = [x / total for x in lista]
    real_lista.append(lista[topic1])
  
    color_list = []
    widths = []
  for i in range(0,13):
    color_list.append(cols[topic1])
    widths.append(0.5)
  fw1.add_trace(go.Bar(y = names, x = real_lista, marker_color=color_list, orientation='h',width=widths, name=topics[topic1]))

fw1.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',barmode='stack',legend={'traceorder':'normal'})
fw1.update_yaxes(visible=True, linecolor='black')
fw1.update_xaxes(visible=True, linecolor='black', gridcolor='grey',tickformat=".0%")
fw1['layout']['xaxis'].update(title='',range=[0,1], autorange=False)
fw1.show()

"""Post Covid"""

#@title

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fw2 = go.Figure()

names = ["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"]
topics = ["Education","Employment", "Faculty", "Research", "Health", "Society"]

for topic1 in range(0,6):
  real_lista = []
  for string in sorted(poscov_en['user.name'].unique()):
    temp = poscov_en[poscov_en['user.name'] == string ]
    lista = []
    for topic2 in range(0,6):
      count = len(temp[temp['topic'] == topic2].index)
      lista.append(count)
    total = sum(lista)
    if total!= 0: lista[:] = [x / total for x in lista]
    real_lista.append(lista[topic1])

    widths = []
    color_list = []
  for i in range(0,13):
    color_list.append(cols[topic1])
    widths.append(0.5)
  fw2.add_trace(go.Bar(y = names, x = real_lista, marker_color=color_list, orientation='h',width=0.5, name=topics[topic1]))

fw2.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',barmode='stack',legend={'traceorder':'normal'})
fw2.update_yaxes(visible=True, linecolor='black')
fw2.update_xaxes(visible=True, linecolor='black', gridcolor='grey',tickformat=".0%")
fw2['layout']['xaxis'].update(title='',range=[0,1], autorange=False)
fw2.show()

lista_precovid = []
lista_poscovid = []

for string in sorted(precov_en['user.name'].unique()):
  temp = precov_en[precov_en['user.name'] == string ]
  mavg = round(temp[['created_at','text']].resample('M', on='created_at').count()['text'].mean(),2)
  lista_precovid.append(mavg)

  temp = poscov_en[poscov_en['user.name'] == string ]
  mavg = round(temp[['created_at','text']].resample('M', on='created_at').count()['text'].mean(),2)
  lista_poscovid.append(mavg)


temp1 = precov_en.groupby('user.name')['favorite_count','retweet_count'].agg(['mean']).reset_index()
temp1.columns=['user.name','Favorite Avg','Retweet Avg']
temp1['Favorite Avg'] = temp1['Favorite Avg'].round(2)
temp1['Retweet Avg'] = temp1['Retweet Avg'].round(2)
temp2 = poscov_en.groupby('user.name')['favorite_count','retweet_count'].agg(['mean']).reset_index()
temp2.columns=['user.name','Covid-19 Favorite Avg','Covid-19 Retweet Avg']
temp2['Covid-19 Favorite Avg'] = temp2['Covid-19 Favorite Avg'].round(2)
temp2['Covid-19 Retweet Avg'] = temp2['Covid-19 Retweet Avg'].round(2)
temp = pd.merge(temp1, temp2, on='user.name', how='outer')
temp['Monthly Post Avg'] = lista_precovid
temp['Covid-19 Monthly Post Avg'] = lista_poscovid
temp = temp[['user.name','Favorite Avg','Covid-19 Favorite Avg','Retweet Avg','Covid-19 Retweet Avg','Monthly Post Avg','Covid-19 Monthly Post Avg']]
temp

from textblob import TextBlob

def getPolarity2(row):
    try:
      return TextBlob(row['text']).sentiment.polarity
    except:
      return 0

def getSentiment(row):
  if row['polarity'] > 0.33:
    val = 'Positive'
  elif row['polarity'] < -0.33:
    val = 'Negative'
  else:
    val = 'Neutral'
  return val

df['polarity'] = df.apply(getPolarity2, axis=1)
precov_en['polarity'] = precov_en.apply(getPolarity2, axis=1)
precov_en['sentiment'] = precov_en.apply(getSentiment, axis=1)
poscov_en['polarity'] = poscov_en.apply(getPolarity2, axis=1)
poscov_en['sentiment'] = poscov_en.apply(getSentiment, axis=1)

colors = ['red','grey','green'] 

fig = make_subplots(rows=1,cols=7,subplot_titles=("Total","Education","Employment","Faculty","Research","Health","Society"))
df['topic'] = df['topic'].astype(int)

temp1 = precov_en.groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp2 = precov_en[precov_en['topic'] == 0].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp3 = precov_en[precov_en['topic'] == 1].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp4 = precov_en[precov_en['topic'] == 2].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp5 = precov_en[precov_en['topic'] == 3].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp6 = precov_en[precov_en['topic'] == 4].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp7 = precov_en[precov_en['topic'] == 5].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()

fig.add_trace(go.Bar(x=temp1['sentiment'], y=temp1['count']/temp1['count'].sum(), marker_color=colors), row=1, col=1)
fig.add_trace(go.Bar(x=temp2['sentiment'], y=temp2['count']/temp2['count'].sum(), marker_color=colors), row=1, col=2)
fig.add_trace(go.Bar(x=temp3['sentiment'], y=temp3['count']/temp3['count'].sum(), marker_color=colors), row=1, col=3)
fig.add_trace(go.Bar(x=temp4['sentiment'], y=temp4['count']/temp4['count'].sum(), marker_color=colors), row=1, col=4)
fig.add_trace(go.Bar(x=temp5['sentiment'], y=temp5['count']/temp5['count'].sum(), marker_color=colors), row=1, col=5)
fig.add_trace(go.Bar(x=temp6['sentiment'], y=temp6['count']/temp6['count'].sum(), marker_color=colors), row=1, col=6)
fig.add_trace(go.Bar(x=temp6['sentiment'], y=temp7['count']/temp7['count'].sum(), marker_color=colors), row=1, col=7)
fig.update_layout(showlegend=False)

fig.update_layout( title={'text': "",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top', 'font':{'size': 18}},font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(visible=True, linecolor='black')
fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey',tickformat=".0%")

for i in range(1,8):
  fig['layout']['yaxis'+str(i)].update(title='',range=[0,0.9], autorange=False, )

fig.update_annotations(font_size=20)
fig.show()

colors = ['red','grey','green'] 

fig = make_subplots(rows=1,cols=7,subplot_titles=("Total","Education","Employment","Faculty","Research","Health","Society"))
df['topic'] = df['topic'].astype(int)

temp1 = poscov_en.groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp2 = poscov_en[poscov_en['topic'] == 0].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp3 = poscov_en[poscov_en['topic'] == 1].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp4 = poscov_en[poscov_en['topic'] == 2].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp5 = poscov_en[poscov_en['topic'] == 3].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp6 = poscov_en[poscov_en['topic'] == 4].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
temp7 = poscov_en[poscov_en['topic'] == 5].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()

fig.add_trace(go.Bar(x=temp1['sentiment'], y=temp1['count']/temp1['count'].sum(), marker_color=colors), row=1, col=1)
fig.add_trace(go.Bar(x=temp2['sentiment'], y=temp2['count']/temp2['count'].sum(), marker_color=colors), row=1, col=2)
fig.add_trace(go.Bar(x=temp3['sentiment'], y=temp3['count']/temp3['count'].sum(), marker_color=colors), row=1, col=3)
fig.add_trace(go.Bar(x=temp4['sentiment'], y=temp4['count']/temp4['count'].sum(), marker_color=colors), row=1, col=4)
fig.add_trace(go.Bar(x=temp5['sentiment'], y=temp5['count']/temp5['count'].sum(), marker_color=colors), row=1, col=5)
fig.add_trace(go.Bar(x=temp6['sentiment'], y=temp6['count']/temp6['count'].sum(), marker_color=colors), row=1, col=6)
fig.add_trace(go.Bar(x=temp7['sentiment'], y=temp7['count']/temp7['count'].sum(), marker_color=colors), row=1, col=7)
fig.update_layout(showlegend=False)

fig.update_layout( title={'text': "",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top', 'font':{'size': 18}},font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(visible=True, linecolor='black')
fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey',tickformat=".0%")

for i in range(1,8):
  fig['layout']['yaxis'+str(i)].update(title='',range=[0,0.9], autorange=False, )

fig.update_annotations(font_size=20)
fig.show()


#@title
import plotly.figure_factory as ff
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

fig = make_subplots(rows=1,cols=6,subplot_titles=("Education","Employment","Faculty","Research","Health","Society"))
colors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
topics = ["Education","Employment","Faculty","Research","Health","Society"]

cs = 1
for j in range(0,6):
  hist_data = []
  hist_data.append(df[df['topic'] == j]["polarity"].tolist())
  group_label = [topics[j]]
  temp_fig = ff.create_distplot(hist_data, group_label,bin_size=.05)
  fig.add_trace(go.Histogram(temp_fig['data'][0],marker_color=colors[j]), row=1, col=cs)
  fig.add_trace(go.Scatter(temp_fig['data'][1],line=dict(color=colors[j], width=2)), row=1, col=cs)
  cs += 1
  i += 1
fig.update_layout(showlegend=False,height=600)

for i in range(1,7):
  fig['layout']['xaxis'+str(i)].update(title='', range=[-1,1], autorange=False)

fig.show()

df.to_csv('df_en2.csv', index=False)