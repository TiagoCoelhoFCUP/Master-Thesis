# -*- coding: utf-8 -*-

#Carregar os pacotes necessarios
import pandas as pd  
import numpy as np   
import json 
import emoji
from matplotlib import pyplot

import warnings
warnings.filterwarnings('ignore')

"""Começo por fazer a leitura dos JSON para uma dataframe do Pandas, mas como pode ver, existem campos onde o valor são objetos JSON ainda, então pensei em fazer uma normalização."""
df = pd.read_json('top_u2020.json',lines=True)
# To display the top 5 rows
df.head(5)

"""Carregar e concater á dataframe existe as instituições que faltavam recolher."""
df_2 = pd.read_json('top_u2020_2.json',lines=True)
df = pd.concat([df, df_2],ignore_index=True)

"""Mesmo após a normalização, existem ainda alguns campos como o "entities.hashtags" ou o 	"entities.symbols" que possuem listas de JSONs e fazem com que o dataframe não seja flat. Se calhar valeriam a pena normalizar também, mas acabei por prosseguir com a dataframe desta forma."""
json_struct = json.loads(df.to_json(orient="records"))    
df_flat = pd.json_normalize(json_struct) #use pd.io.json
df_flat['created_at'] = df['created_at']
df_flat['id_str'] = df_flat['id.$numberLong']
df_flat.head(5)


"""De seguida tratei de identificar as instituições envolvidas, a sua localização e o seu screen name.
Por alguma razão a Universidade de Oxford tinha 2 nomes associados á mesma, por isso, por uma questão de uniformidade, atribuí a todos os tweets da mesma um nome comum.

"""
df_flat['user.name'] = df_flat['user.name'].replace(['Oxford University'],'University of Oxford')
df_temp = df_flat[['user.name', 'user.screen_name', 'user.location']].drop_duplicates()
locations = df_temp['user.location']
orgs = df_temp['user.name']
screen_names = df_temp['user.screen_name']

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['Organization', 'Screen Name','Locations']),cells=dict(values=[orgs, screen_names, locations]))])
fig.show()

"""Seguidamente tentei obter uma informação geral das métricas que, a meu ver, identificam melhor o engagement: "retweet_count" e "favorite_count"""
df_new = df_flat[['id_str', '_id.$oid',	 'created_at', 'text',	'truncated', 'source', 'is_quote_status',	'retweet_count' ,'favorite_count', 'favorited',	'retweeted', 'retweeted_status.text', 'quoted_status.text', 'lang', 'user.name',	'user.screen_name', 'user.location',	'user.description',	'user.followers_count', 'user.statuses_count']]
excluded = ['Penn','Princeton University', 'Columbia University','Universidade Porto']
df_new = df_new[~df_new['user.name'].isin(excluded)]
df_new = df_new[df_new['created_at'] <= '2020-09-01 00:00:00']
df_new = df_new[df_new['created_at'] >='2019-09-01 00:00:00']
df_new[['retweet_count','favorite_count']].describe()

"""Após isso quis identificar a percentagem de tweets correspondentes a cada organização, no total dos 46 429. Acaba por ser uma distribuição bem equilibrada.

"""
df_temp = df_new.groupby(['user.screen_name']).size().reset_index(name='counts')

import plotly.express as px

fig = px.bar(df_temp, x='user.screen_name', y='counts',title="Volume of tweets",labels={'user.screen_name':'Institution','counts':'Number of Tweets'})
fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1000)
fig.show()

df_temp1 = df_new.groupby(['user.screen_name']).size().reset_index(name='Publications').sort_values(by=['user.screen_name'], ascending=True)
df_temp2 = df_new.groupby(['user.screen_name'])[['user.followers_count']].mean().reset_index().sort_values(by=['user.screen_name'], ascending=True)
df_temp= df_temp1.merge(df_temp2, left_index=True, right_index=True, how='outer',suffixes=('', '_y'))
df_temp.drop(df_temp.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
df_temp

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_temp.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df_temp.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
])
fig.show()

"""Tentei também obter dados estatisticos sobre as métricas previamente mencionadas para cada instituição em específico."""
df_temp = df_new[['user.name','retweet_count', 'favorite_count']].groupby(['user.name']).describe()

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_temp.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df_temp.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
])
fig.show()

"""Criação de boxplots para cada instituição, referentes ao número de retweets e número de favoritos (Foi preciso inicialmente remover os outliers dos dados de cada instituição, para tornar possivel a visualização desses dados em boxplots sucessivos)."""
cols = ['favorite_count', 'retweet_count'] #outlier columns

lista = []
for string in df_new['user.name'].unique():
  df_temp = df_new[df_new['user.name'] == string ]
  Q1 = df_temp[cols].quantile(0.25)
  Q3 = df_temp[cols].quantile(0.75)
  IQR = Q3 - Q1
  df_temp = df_temp[~((df_temp[cols] < (Q1 - 1.5 * IQR)) |(df_temp[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
  lista.append(df_temp)

df_new_no_outliers = pd.concat(lista)  

import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

for string in df_new['user.name'].unique():
  fig.add_trace(go.Box(x=df_new_no_outliers[df_new_no_outliers['user.name'] == string ]['favorite_count'], boxpoints=False, name=string))

fig.update_layout(xaxis_title='Number of Favorites',showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_layout(font=dict(size=15), width=800, height=500)
fig.show()

fig = go.Figure()

for string in df_new['user.name'].unique():
  fig.add_trace(go.Box(x=df_new_no_outliers[df_new_no_outliers['user.name'] == string ]['retweet_count'], boxpoints=False, name=string))

fig.update_layout(xaxis_title='Number of Retweets',showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_layout(font=dict(size=15), width=800, height=500)
fig.show()

"""Dados estatísticos sobre as métricas de engagement após a remoção dos outliers."""

df_temp = df_new_no_outliers[['user.name','retweet_count', 'favorite_count']].groupby(['user.name']).describe()

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_temp.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df_temp.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
])
fig.show()

temp = df_new.groupby(['user.screen_name']).mean('user.followers_count').reset_index()
temp

import plotly.express as px

fig = px.bar(temp, x='user.screen_name', y='user.followers_count',title="Followers per HEI",labels={'user.screen_name':'Institution','user.followers_count':'Number of Followers'})
fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1000)
fig.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
temp = df_new.groupby(['user.name']).size().reset_index(name='counts')
temp2 = df_new_no_outliers[['user.name','retweet_count', 'favorite_count','user.followers_count']].groupby(['user.name']).mean().reset_index()
temp = temp.merge(temp2, on='user.name', how='left')
temp['eficiency'] = temp['retweet_count']/temp['counts']/temp['user.followers_count']
temp['eficiency'] = scaler.fit_transform(temp['eficiency'].values.reshape(-1,1))
temp

"""Por fim, tentei observar a possivel existênica de alguma corelação entre o número de seguidores da instituição que publicou o tweet e o número de favoritos que esse tweet obteve. 
Para meu espanto essa relação parece não existir. Uma coisa que támbem me surpreendeu é a pouca variação do número de inscritos das instituições em causa, que pode ser o motivo de não se observar uma correlação entre o numero de inscritos e o numero de favoritos: Os dados foram recolhidos num curto espaço de tempo (?)
"""

fig = px.scatter(df_new, x="retweet_count", y="user.followers_count", color="user.name", log_x=True, labels={
                     "retweet_count": "Tweet's Retweet Count",
                     "user.followers_count": "Institution Followers Count",
                     "user.name": "Institution"
                 })
fig.update_layout(font=dict(size=20),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1000,showlegend=True)
fig.update_xaxes(visible=True, linecolor='black')
fig.update_yaxes(visible=True, linecolor='black')
fig.show()

"""Remoção de tweets duplicados pelo "id_str"(nenhum duplicado)."""
df_new.drop_duplicates(subset=['id_str'])

"""Remoção dos tweets duplicados pelo "_id.$oid"	(nenhum duplicado, são ambos identificadores únicos. Decidi então utilizar o "id_str"."""
df_new.drop_duplicates(subset=['_id.$oid'])
df_new.drop(columns=['_id.$oid'])

"""Criração de uma tabela com o número de tweets, data do tweet mais antigo e data to tweet mais recente, para cada HEI."""
hei_stats = df_new.groupby(['user.name'])['created_at'].agg(['count','min', 'max']).reset_index()
hei_stats.sort_values('count', ascending=False, ignore_index=True)
hei_stats['min'] = hei_stats['min'].dt.strftime('%Y-%m-%d')
hei_stats['max'] = hei_stats['max'].dt.strftime('%Y-%m-%d')
hei_stats

"""Criação de um gráfico de linhas para observar o período de atividade para cada instituição."""
#@title
lista = []
conta = 1;
for string in df_new['user.name'].unique():
  temp = df_new[df_new['user.name'] == string ]
  temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['created_at'].agg(['count']).reset_index()
  temp['user.name'] = string
  temp['data'] = np.where(temp['count']> 0, conta, 0)
  conta += 1
  lista.append(temp)

result = pd.concat(lista)

fig = px.line(result, x="created_at", y="data", color='user.name', hover_data={'data':False}, labels={
                     "created_at": "Month/Year",
                     "user.name": "Institution", 
                 },
                 title="Activity Period",width=3000)

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")

fig.update_yaxes(visible=False, showticklabels=False)


fig.show()

"""Criação de um gráfico de linhas com a distribuição mensal de posts para cada instituição."""

#@title
lista = []
for string in df_new['user.name'].unique():
 temp = df_new[df_new['user.name'] == string ]
 temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['created_at'].agg(['count']).reset_index()
 temp['user.name'] = string
 lista.append(temp)

result = pd.concat(lista)

fig = px.line(result, x="created_at", y="count", color='user.name',labels={
                     "created_at": "Tweet's Month/Year of Emission",
                     "count": "Number of Tweets",
                     "user.name": "Institution"
                 },
                 title="",width=1000)

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")

fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(visible=True, linecolor='black')
fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

fig.show()

"""Criação de um gráfico de linhas com a distribuição mensal de favoritos para cada instituição."""

#@title
lista = []
for string in df_new['user.name'].unique():
  temp = df_new[df_new['user.name'] == string ]
  temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['favorite_count'].agg(['sum','count']).reset_index()
  temp['fav_per_tweet'] = temp['sum']/temp['count']
  temp['user.name'] = string
  lista.append(temp)

result = pd.concat(lista)

fig = px.line(result, x="created_at", y="fav_per_tweet", color='user.name',labels={
                     "created_at": "Tweet's Month/Year of Emission",
                     "fav_per_tweet": "Number of Favorites/Tweet",
                     "user.name": "Institution"
                 },
                 title="",width=1000)

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")

fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(visible=True, linecolor='black')
fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')
fig.show()

"""Criação de um gráfico de linhas com a distribuição mensal de retweets para cada instituição."""

#@title
lista = []
for string in df_new['user.name'].unique():
  temp = df_new[df_new['user.name'] == string ]
  temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['retweet_count'].agg(['sum','count']).reset_index()
  temp['ret_per_tweet'] = temp['sum']/temp['count']
  temp['user.name'] = string
  lista.append(temp)

result = pd.concat(lista)

fig = px.line(result, x="created_at", y="ret_per_tweet", color='user.name',labels={
                     "created_at": "Tweet's Month/Year of Emission",
                     "ret_per_tweet": "Number of Retweets/Tweet",
                     "user.name": "Institution"
                 },
                 title="Monthtly Retweet Distribution",width=3000)

fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")

'fig.update_layout(font=dict(size=20))'

fig.show()

"""Criação de um gráfico de linhas com a distribuição trimestral de posts para cada instituição."""

#@title
lista = []
for string in df_new['user.name'].unique():
  temp = df_new[df_new['user.name'] == string ]
  temp = temp.groupby(pd.Grouper(key='created_at',freq='Q'))['created_at'].agg(['count']).reset_index()
  temp['user.name'] = string
  lista.append(temp)

result = pd.concat(lista)

fig = px.line(result, x="created_at", y="count", color='user.name',labels={
                     "created_at": "Tweet's Quarter/Year of Emission",
                     "count": "Number of Tweets",
                     "user.name": "Institution"
                 },
                 title="Quarterly Post Distribution",width=3000,height=500)

fig.update_xaxes(
    dtick="M3",
    tickformat="%qº Trimestre\n%Y")

fig.show()

"""Criação de um heat map Dia_da_semana x Hora_do_dia do número de posts, para a Universidade de Oxford no mês de Agosto de 2019 (Escolhida esta instituição e este mês por ser a instância com o maior número de tweets)."""

temp = df_new[df_new['user.name'] == 'Massachusetts Institute of Technology (MIT)' ]
temp = temp[temp['created_at'] <= '2020-09-01 00:00:00']
temp = temp[temp['created_at'] >='2019-09-01 00:00:00']
temp['year'] = pd.DatetimeIndex(temp['created_at']).year
temp['month'] = pd.DatetimeIndex(temp['created_at']).month
temp['day'] = temp['created_at'].dt.dayofweek
#temp = temp[(temp['year'] == 2019) & (temp['month'] == 8)]
temp['hour'] = pd.DatetimeIndex(temp['created_at']).hour
temp = temp[['day','hour']]
temp = temp.groupby(['day','hour'])['hour'].agg(['count']).reset_index()

for i in range(0,7):
  for j in range(0,24):
    if not ((temp['day'] == i) & (temp['hour'] == j)).any():
      row = [i, j, 0]
      temp.loc[len(temp)] = row
  

temp = temp.pivot_table(index='day', columns='hour', values='count')
temp

fig = px.imshow(temp,
                labels=dict(x="Time of Day(Hours)", y="Day of Week", color="Number of Tweets"),title = "MIT",
                y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']) 

fig.update_xaxes(side="top")
fig.update_layout(font=dict(size=15),width=800)
fig.show()

"""Heatmaps de todas as instituições com a distribuição de posts por hora x dia da semana."""

#@title
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
order = ['Sunday','Saturday','Friday','Thursday','Wednesday','Tuesday','Monday']

def week_heatmap(name):
  temp = df_new[df_new['user.name'] == name ]
  temp['day'] = temp['created_at'].dt.dayofweek
  temp['hour'] = pd.DatetimeIndex(temp['created_at']).hour
  temp = temp[['day','hour']]
  temp = temp.groupby(['day','hour'])['hour'].agg(['count']).reset_index()

  for i in range(0,7):
    for j in range(0,24):
      if not ((temp['day'] == i) & (temp['hour'] == j)).any():
        row = [i, j, 0]
        temp.loc[len(temp)] = row
  
  temp['day'] = temp['day'].map(dayOfWeek)
  temp = temp.pivot_table(index='day', columns='hour', values='count')
  temp = temp.reindex(index = order)
  return temp

fig = make_subplots(rows=3,cols=4,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UCL","University of Oxford","UC Berkeley"))

fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Caltech")),coloraxis = "coloraxis"),
    row=1, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Cambridge University")),coloraxis = "coloraxis"),
    row=1, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("ETH Zurich")),coloraxis = "coloraxis"),
    row=1, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Harvard University")),coloraxis = "coloraxis"),
    row=1, col=4
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Imperial College")),coloraxis = "coloraxis"),
    row=2, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Johns Hopkins University")),coloraxis = "coloraxis"),
    row=2, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Massachusetts Institute of Technology (MIT)")),coloraxis = "coloraxis"),
    row=2, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("Stanford University")),coloraxis = "coloraxis"),
    row=2, col=4
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("The University of Chicago")),coloraxis = "coloraxis"),
    row=3, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("UCL")),coloraxis = "coloraxis"),
    row=3, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("University of Oxford")),coloraxis = "coloraxis"),
    row=3, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(week_heatmap("UC Berkeley")),coloraxis = "coloraxis"),
    row=3, col=4
)

fig.update_yaxes(categoryorder='array', categoryarray= order)
fig.update_layout(height=700, width=1300, coloraxis = {'colorscale':'plasma'},font=dict(family="Times New Roman",size=12,color="black"))
fig.show()

"""Heatmaps de todas as instituições com a distribuição de posts por dia da semana x mês."""

#@title
order_month = ['January','February','March','April','May','June','July','August','September','October','November','December']
monthOfYear={1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

def month_day_heatmap(name):
  temp = df_new[df_new['user.name'] == name ]
  temp['day'] = temp['created_at'].dt.dayofweek
  temp['month'] = pd.DatetimeIndex(temp['created_at']).month
  temp = temp[['month','day']]
  temp = temp.groupby(['day','month'])['month'].agg(['count']).reset_index()

  for i in range(0,7):
    for j in range(1,13):
      if not ((temp['day'] == i) & (temp['month'] == j)).any():
        row = [i, j, 0]
        temp.loc[len(temp)] = row
  
  temp['day'] = temp['day'].map(dayOfWeek)
  temp['month'] = temp['month'].map(monthOfYear)
  temp = temp.pivot_table(index='day', columns='month', values='count')
  temp = temp.reindex(index = order)
  temp = temp.reindex(order_month, axis=1)
  return temp

fig = make_subplots(rows=3,cols=4,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UCL","University of Oxford","UC Berkeley"))

fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Caltech")),coloraxis = "coloraxis"),
    row=1, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Cambridge University")),coloraxis = "coloraxis"),
    row=1, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("ETH Zurich")),coloraxis = "coloraxis"),
    row=1, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Harvard University")),coloraxis = "coloraxis"),
    row=1, col=4
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Imperial College")),coloraxis = "coloraxis"),
    row=2, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Johns Hopkins University")),coloraxis = "coloraxis"),
    row=2, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Massachusetts Institute of Technology (MIT)")),coloraxis = "coloraxis"),
    row=2, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("Stanford University")),coloraxis = "coloraxis"),
    row=2, col=4
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("The University of Chicago")),coloraxis = "coloraxis"),
    row=3, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("UCL")),coloraxis = "coloraxis"),
    row=3, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("University of Oxford")),coloraxis = "coloraxis"),
    row=3, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_day_heatmap("UC Berkeley")),coloraxis = "coloraxis"),
    row=3, col=4
)

fig.update_yaxes(categoryorder='array', categoryarray= order)
fig.update_layout(height=700, width=1300, coloraxis = {'colorscale':'plasma'},font=dict(family="Times New Roman",size=12,color="black"))
fig.show()

"""Heatmaps de todas as instituições com a distibuição de posts por hora do dia x mês.

"""

#@title
def month_hour_heatmap(name):
  temp = df_new[df_new['user.name'] == name ]
  temp['hour'] = pd.DatetimeIndex(temp['created_at']).hour
  temp['month'] = pd.DatetimeIndex(temp['created_at']).month
  temp = temp[['month','hour']]
  temp = temp.groupby(['hour','month'])['month'].agg(['count']).reset_index()

  for i in range(0,24):
    for j in range(1,13):
      if not ((temp['hour'] == i) & (temp['month'] == j)).any():
        row = [i, j, 0]
        temp.loc[len(temp)] = row
  
  temp['month'] = temp['month'].map(monthOfYear)
  temp = temp.pivot_table(index='hour', columns='month', values='count')
  temp = temp.reindex(order_month, axis=1)
  return temp

fig = make_subplots(rows=3,cols=4,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UCL","University of Oxford","UC Berkeley"))

fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Caltech")),coloraxis = "coloraxis"),
    row=1, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Cambridge University")),coloraxis = "coloraxis"),
    row=1, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("ETH Zurich")),coloraxis = "coloraxis"),
    row=1, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Harvard University")),coloraxis = "coloraxis"),
    row=1, col=4
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Imperial College")),coloraxis = "coloraxis"),
    row=2, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Johns Hopkins University")),coloraxis = "coloraxis"),
    row=2, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Massachusetts Institute of Technology (MIT)")),coloraxis = "coloraxis"),
    row=2, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("Stanford University")),coloraxis = "coloraxis"),
    row=2, col=4
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("The University of Chicago")),coloraxis = "coloraxis"),
    row=3, col=1
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("UCL")),coloraxis = "coloraxis"),
    row=3, col=2
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("University of Oxford")),coloraxis = "coloraxis"),
    row=3, col=3
)
fig.add_trace(
    go.Heatmap(df_to_plotly(month_hour_heatmap("UC Berkeley")),coloraxis = "coloraxis"),
    row=3, col=4
)

fig.update_yaxes(categoryorder='array', categoryarray= order)
fig.update_layout(height=700, width=1300, coloraxis = {'colorscale':'plasma'},font=dict(family="Times New Roman",size=12,color="black"))
fig.show()
