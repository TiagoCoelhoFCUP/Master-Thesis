if __name__ == '__main__': 

    import nltk; nltk.download('stopwords')
    import re
    import emoji
    import numpy as np
    import pandas as pd
    from pprint import pprint
    seed = 15 

    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # spacy for lemmatization
    import spacy

    # Plotting tools
    import pyLDAvis
    import pyLDAvis.gensim  
    import matplotlib.pyplot as plt
    # %matplotlib inline

    # Enable logging for gensim - optional
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

    import warnings
    warnings.filterwarnings('ignore')

    """Carreagar e concatenar os dois conjuntos de tweets."""
    df_full = pd.read_csv('full_tweets.csv',index_col=0, encoding = "UTF-8")
    df_full['created_at'] = pd.to_datetime(df_full.created_at)
    df_full2 = pd.read_csv('full_tweets2.csv', encoding = "UTF-8")
    df_full2['created_at'] = pd.to_datetime(df_full2.created_at)
    df_new = pd.concat([df_full, df_full2],ignore_index=True)
    excluded = ['Penn','Princeton University', 'Columbia University']
    df_new = df_new[~df_new['user.name'].isin(excluded)]
    df_new = df_new[df_new['created_at'] <= '2020-09-01 00:00:00']
    df_new = df_new[df_new['created_at'] >='2019-09-01 00:00:00']

    """Limpeza de texto."""
    def alter_emoji(text):
        return emoji.demojize(text).replace(":"," ")

    def cleanTxt(row):
      val = re.sub(r'@[A-Za-z0-9_]+:* *', '', row['text'])
      val = re.sub(r'RT[\s]+', '', val)
      val = re.sub(r'http?s:\/\/\S+', '', val)
      val = re.sub(r'&amp;', '&', val)
      val = re.sub(r'&lt;', '<', val)
      val = re.sub(r'&gt;', '>', val)
      val = re.sub(r'#', '', val)
      val = re.sub(r'\?{2,}',' multipleQuestion',val)
      val = re.sub(r'\!{2,}',' multipleExclamation',val)
      val = re.sub(r'\.{2,}',' multiplePeriods',val)
      val = re.sub(r'\s+', ' ', val)
      val = re.sub(r"\’", "", val)
      #val = re.sub(r'#(\w+)', '', val)
      val = alter_emoji(val)
      return val

    df_new['text'] = df_new.apply(cleanTxt, axis=1)

    """Simple pré-processamento "built in" do gensim."""
    data_en = df_new[df_new['user.name'] != 'Universidade Porto'].text.values.tolist()

    def tweet_to_words(tweets):
        for tweet in tweets:
            yield(gensim.utils.simple_preprocess(str(tweet), deacc=True))  # deacc=True removes punctuations

    data_words_en = list(tweet_to_words(data_en))

    """Construção dos Bigrams e Trigrams (min_count diferente para o modelo PT e EN devido ás diferentes dimensões dos datasets). """
    # Build the bigram and trigram models for EN language
    bigram_en = gensim.models.Phrases(data_words_en, min_count=50, threshold=100) 
    trigram_en = gensim.models.Phrases(bigram_en[data_words_en], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod_en = gensim.models.phrases.Phraser(bigram_en)
    trigram_mod_en = gensim.models.phrases.Phraser(trigram_en)

    """Remoção de stopwords e lematização."""
    from nltk.corpus import stopwords
    stop_en = stopwords.words('english')

    import en_core_web_sm

    nlp_en = en_core_web_sm.load(disable=['parser', 'ner'])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts,lang):
        stop_words = stop_en
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts,lang):
        bigram_mod = bigram_mod_en
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts,lang):
        trigram_mod = trigram_mod_en
        bigram_mod = bigram_mod_en
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, lang,  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        nlp = nlp_en
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_.lower() for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_en_nostops = remove_stopwords(data_words_en,'en')

    # Form Trigrams
    data_words_trigrams_en = make_trigrams(data_words_en_nostops,'en')

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized_en = lemmatization(data_words_trigrams_en, 'en', allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    """Criaçãod de dicionários, corpus e TF-IDF."""
    # Create Dictionaries
    id2word_en = corpora.Dictionary(data_lemmatized_en)
    #id2word_en.save("id2worden")
    #id2word_en = corpora.Dictionary.load("id2worden")

    # Create Corpus
    texts_en = data_lemmatized_en

    # Term Document Frequency
    corpus_en = [id2word_en.doc2bow(text) for text in texts_en]

    # Human readable format of corpus (term-frequency)
    print([[(id2word_en[id], freq) for id, freq in cp] for cp in corpus_en[:1]])

    """Criação de modelos LDA Mallet (usa algoritmo diferente, produz possivelmente tópicos de melhor qualidade)."""
    """Para determinar o número ideal de tópicos, cria-se um conjunto de modelos com números diferentes de tópicos e calculam-se os seus respetivos valores de coherence. O modelo ideal será o que possui maior coherence."""

    file1 = open('mallet_path.txt', 'r')
    line = file1.readline()
    import os
    os.environ['MALLET_HOME'] = line
    mallet_path = line +'\\bin\\mallet'

    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        np.random.seed(seed)
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, workers=1, id2word=dictionary, iterations=3000, random_seed=seed)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    model_list_LDA_en, coherence_values_LDA_en = compute_coherence_values(dictionary=id2word_en, corpus=corpus_en, texts=data_lemmatized_en, start=2, limit=10, step=1)

    import gensim.models.nmf as nmf

    def compute_coherence_values2(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        np.random.seed(seed)
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = nmf.Nmf(corpus=corpus, num_topics=num_topics, id2word=dictionary, w_max_iter=3000, random_state=seed)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    model_list_NMF_en, coherence_values_NFM_en = compute_coherence_values2(dictionary=id2word_en, corpus=corpus_en, texts=data_lemmatized_en, start=2, limit=10, step=1)


    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = [2,3,4,5,6,7,8,9,10], y = coherence_values_NFM_en, name="NMF",line=dict(width=5)))
    fig.add_trace(go.Scatter(x = [2,3,4,5,6,7,8,9,10], y = coherence_values_LDA_en, name="LDA",line=dict(width=5)))

    fig.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title_font=dict(size=25), visible=True, linecolor='black',title="Number of Topics")
    fig.update_yaxes(title_font=dict(size=25), visible=True, linecolor='black', gridcolor='grey', title="Coherence")
    fig.show()

    optimal_model_LDA_en = model_list_LDA_en[3]
    optimal_model_NMF_en = model_list_NMF_en[3]

    """EN LDA Model - 5 Topics."""

    #@title
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.colors as mcolors

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_en,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = optimal_model_LDA_en.show_topics(formatted=False)

    fig, axes = plt.subplots(1, 5, figsize=(20,8), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    """EN NMF Model - 5 Topics"""

    #@title
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.colors as mcolors

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_en,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = optimal_model_NMF_en.show_topics(formatted=False)

    fig, axes = plt.subplots(1, 5, figsize=(20,8), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout() 
    plt.show()


    """Como os modelos LDA possuiam scores de coherence maiores e produzem tópicos de melhor qualidade, decidi utiliza-los nos próximos passos. Criei uma dataframe que associa a cada tweet o seu tópico dominante, as suas keywords e a percentagem do tweet dominada por esse tópico."""

    def format_topics_sentences(ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            topics = {}
            dominant = -1
            dominant_perc = -1
            topic_keywords = ""
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    dominant = topic_num
                    wp = ldamodel.show_topic(dominant)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    dominant_perc = prop_topic
                    topics[dominant] = prop_topic
                else:
                    topics[topic_num]  = prop_topic

            count = 0
            for value in topics.values():
              if value >= 0.2:
                count+=1

            sent_topics_df = sent_topics_df.append(pd.Series([int(dominant), round(dominant_perc,4), topic_keywords, round(topics[0],4), round(topics[1],4), round(topics[2],4), round(topics[3],4), round(topics[4],4), count]), ignore_index=True)
                    
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords','Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Diversity']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)


    df_topic_sents_keywords_en = format_topics_sentences(ldamodel=optimal_model_LDA_en, corpus=corpus_en, texts=data_en)

    # Format
    df_dominant_topic_en = df_topic_sents_keywords_en.reset_index()
    df_dominant_topic_en.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4','Diversity','Text']

    # Show EN Dataframe
    df_dominant_topic_en.head(10)

    """Obter os tweets mais representativos de cada tópico obtido."""

    sent_topics_sorteddf_mallet_en = pd.DataFrame()
    sent_topics_outdf_grpd_en = df_topic_sents_keywords_en.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd_en:
        sent_topics_sorteddf_mallet_en = pd.concat([sent_topics_sorteddf_mallet_en, 
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet_en.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet_en.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4','Diversity','Text']

    # Show most representative text of each topic in the EN model
    sent_topics_sorteddf_mallet_en.head(10)
    sent_topics_sorteddf_mallet_en.style.set_properties(subset=['Text'], **{'width': '1000px'})
    
    df_temp = sent_topics_sorteddf_mallet_en
    fig = go.Figure(data=[go.Table(header=dict(values=list(df_temp.columns),fill_color='paleturquoise',align='left'),cells=dict(values=df_temp.transpose().values.tolist(),fill_color='lavender',align='left'))])
    fig.show()
    
    """Visualizar os modelos LDA Mallet."""
    # Visualize the topics in EN model 
    #pyLDAvis.enable_notebook(local = True)
    #model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model_LDA_en)
    #vis = pyLDAvis.gensim.prepare(model, corpus_en, dictionary=model.id2word)
    #vis

    """Obter a distribuição de word counts de cada documento (tweet)."""
    doc_lens = [len(d) for d in df_dominant_topic_en.Text]

    # Plot
    plt.figure(figsize=(16,7), dpi=160)
    plt.hist(doc_lens, bins = 1000, color='navy')
    plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,1000,9))
    plt.title('Distribution of EN Document(Tweet) Word Counts', fontdict=dict(size=22))
    plt.show()

    """Obter a distribuição de word counts de cada documento em relação ao seu tópico dominante."""

    #@title
    import seaborn as sns
    import matplotlib.colors as mcolors
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    dic =  {0 : 'Events', 1 : 'Health', 2: 'Research', 3: 'Education', 4: 'Image'}

    fig, axes = plt.subplots(1,5,figsize=(20,6), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):    
        df_dominant_topic_en_sub = df_dominant_topic_en.loc[df_dominant_topic_en.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_en_sub.Text]
        ax.hist(doc_lens, bins = 250, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 500), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i],fontdict=dict(size=15, color=cols[i]))
        ax.set_title(dic[i], fontdict=dict(size=20, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    #plt.xticks(np.linspace(0,1000,9))
    fig.suptitle('', fontsize=30)
    plt.show()

    temp_en = df_new[df_new['user.name'] != 'Universidade Porto']
    temp_en.reset_index(drop=True, inplace=True)

    df_dominant_topic_en.rename(columns={'Dominant_Topic': 'topic', 'Topic_Perc_Contrib': 'topic_perc', 'Topic 0':'topic_0', 'Topic 1':'topic_1', 'Topic 2':'topic_2', 'Topic 3':'topic_3', 'Topic 4':'topic_4', 'Diversity':'diversity'},inplace=True)
    df_dominant_topic_en = df_dominant_topic_en[['topic','topic_perc','topic_0','topic_1','topic_2','topic_3','topic_4','diversity']]

    df_en = pd.merge(temp_en, df_dominant_topic_en, left_index=True, right_index=True)

    df_temp = df_en.groupby(['topic']).size().reset_index(name='counts')

    import plotly.express as px

    fig = px.bar(df_temp, x='topic', y='counts', title="",labels={'topic':'Topic','counts':'Number of Tweets'})
    fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=1440, height=432)
    fig.show()

    #@title

    names = ["Events","Heath","Research","Education","Image"]
    topics = []

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import datetime

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fig = make_subplots(rows=3,cols=4,vertical_spacing=0.1,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"))

    cs = 1
    rs = 1
    for string in sorted(df_en['user.name'].unique()):
      if cs > 4:
        rs += 1
        cs = 1

      lista = []
      temp = df_en[df_en['user.name'] == string ]
      for topic in range(0,5):
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
      topics.append((string,lista))
      cs += 1

    fig.update_xaxes(
        dtick="M4",
        tickformat="%b\n%Y")

    fig.update_layout(height=800,font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black',showticklabels=False)
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

    for i in range(1,13):
      fig['layout']['yaxis'+str(i)].update(title='', range=[0,200], autorange=False)

    for i in range(1,13):
      fig['layout']['xaxis'+str(i)].update(title='', range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)], autorange=False)

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

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        range=[datetime.datetime(2019, 9, 1),datetime.datetime(2020, 9, 1)])

    fig.update_layout(title=string)
    fig.show()

    #@title
    fig = go.Figure()

    string = topics[11][0]
    lista = topics[11][1]
    fig.add_trace(go.Scatter(x=lista[0]['created_at'], y=lista[0]['count'],mode='lines',name='Topic ' + str(0),line=dict(color=cols[0]),legendgroup="0"))
    fig.add_trace(go.Scatter(x=lista[1]['created_at'], y=lista[1]['count'],mode='lines',name='Topic ' + str(1),line=dict(color=cols[1]),legendgroup="1"))
    fig.add_trace(go.Scatter(x=lista[2]['created_at'], y=lista[2]['count'],mode='lines',name='Topic ' + str(2),line=dict(color=cols[2]),legendgroup="2"))
    fig.add_trace(go.Scatter(x=lista[3]['created_at'], y=lista[3]['count'],mode='lines',name='Topic ' + str(3),line=dict(color=cols[3]),legendgroup="3"))
    fig.add_trace(go.Scatter(x=lista[4]['created_at'], y=lista[4]['count'],mode='lines',name='Topic ' + str(4),line=dict(color=cols[4]),legendgroup="4"))

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
    for string in sorted(df_en['user.name'].unique()):
      if cs > 4:
        rs += 1
        cs = 1

      lista = []
      temp = df_en[df_en['user.name'] == string ]
      for topic in range(0,5):
        count = len(temp[temp['topic'] == topic].index)
        lista.append(count)
        
      fig.add_trace(go.Bar(x = ['Events', 'Health', 'Research', 'Education', 'Image'], y = lista, legendgroup="bar",showlegend=False, marker_color=cols[:5]),row=rs,col=cs)
      cs += 1

    fig.update_layout(height=800,font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black')
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

    for i in range(1,13):
      fig['layout']['yaxis'+str(i)].update(title='', range=[0,1000], autorange=False)

    fig.update_annotations(font_size=20)
    fig.show()

    #@title
    precov_en = df_en[df_en['created_at'] < '2020-03-11 00:00:00']

    poscov_en = df_en[df_en['created_at'] >= '2020-03-11 00:00:00']

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import datetime

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fw3 = make_subplots(rows=1,cols=5,shared_yaxes=True,subplot_titles=("Events", "Health", "Research", "Education", "Image"))

    for topic in range(0,5):
      count_pre = len(precov_en[precov_en['topic'] == topic].index)
      count_pos = len(poscov_en[poscov_en['topic'] == topic].index)
      total_pre = len(precov_en.index)
      total_pos = len(poscov_en.index)
      fw3.add_trace(go.Bar(x = ['Pre', 'Post'], y = [count_pre/total_pre, count_pos/total_pos], legendgroup="bar",showlegend=False, marker_color=[cols[topic], cols[topic]]),row=1,col=topic+1)

    fw3.update_annotations(font_size=25)
    fw3.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fw3.update_xaxes(visible=True, linecolor='black')
    fw3.update_yaxes(visible=True, linecolor='black', gridcolor='grey',tickformat="%")

    for i in range(1,6):
      fw3['layout']['yaxis'+str(i)].update(title='',range=[0,0.3], autorange=False, )
    fw3.show()

    #@title
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import datetime

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fw4 = make_subplots(rows=1,cols=5,shared_yaxes=True,subplot_titles=("Events", "Health", "Research", "Education", "Image"))

    tempre_en = precov_en.groupby(['topic'])['retweet_count'].agg(['count','sum'])
    tempos_en = poscov_en.groupby(['topic'])['retweet_count'].agg(['count','sum'])

    for topic in range(0,5):
      fw4.add_trace(go.Bar(x = ['Pre', 'Post'], y = [tempre_en.at[topic,'sum']/tempre_en.at[topic,'count'],tempos_en.at[topic,'sum']/tempos_en.at[topic,'count']], legendgroup="bar",showlegend=False, marker_color=[cols[topic], cols[topic]]),row=1,col=topic+1)

    fw4.update_annotations(font_size=25)
    fw4.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fw4.update_xaxes(visible=True, linecolor='black')
    fw4.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

    for i in range(1,6):
      fw4['layout']['yaxis'+str(i)].update(title='',range=[0,100], autorange=False)
    fw4.show()

    #@title
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import datetime

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fw5 = make_subplots(rows=1,cols=5,shared_yaxes=True,subplot_titles=("Events", "Health", "Research", "Education", "Image"))

    tempre_en = precov_en.groupby(['topic'])['favorite_count'].agg(['count','sum'])
    tempos_en = poscov_en.groupby(['topic'])['favorite_count'].agg(['count','sum'])

    for topic in range(0,5):
      fw5.add_trace(go.Bar(x = ['Pre', 'Post'], y = [tempre_en.at[topic,'sum']/tempre_en.at[topic,'count'],tempos_en.at[topic,'sum']/tempos_en.at[topic,'count']], legendgroup="bar",showlegend=False, marker_color=[cols[topic], cols[topic]]),row=1,col=topic+1)

    fw5.update_annotations(font_size=25)
    fw5.update_layout(font=dict(family="Times New Roman",size=20,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fw5.update_xaxes(visible=True, linecolor='black')
    fw5.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

    for i in range(1,6):
      fw5['layout']['yaxis'+str(i)].update(title='',range=[0,100], autorange=False)
    fw5.show()

    #@title

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import datetime

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fw1 = go.Figure()

    names = ["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"]
    topics = ["Events","Health","Research","Education","Image"]

    for topic1 in range(0,5):
      real_lista = []
      for string in sorted(precov_en['user.name'].unique()):
        temp = precov_en[precov_en['user.name'] == string ]
        lista = []
        for topic2 in range(0,5):
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
    fw1.update_xaxes(visible=True, linecolor='black', gridcolor='grey',tickformat="%")
    fw1['layout']['xaxis'].update(title='',range=[0,1], autorange=False)
    fw1.show()

    #@title

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import datetime

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fw2 = go.Figure()

    names = ["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"]
    topics = ["Events","Health","Research","Education","Image"]

    for topic1 in range(0,5):
      real_lista = []
      for string in sorted(poscov_en['user.name'].unique()):
        temp = poscov_en[poscov_en['user.name'] == string ]
        lista = []
        for topic2 in range(0,5):
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
    fw2.update_xaxes(visible=True, linecolor='black', gridcolor='grey',tickformat="%")
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

    """Pre-Covid19"""
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

    def normalizePolarity(row):
      if row['polarity'] > 0:
        val = 1
      elif row['polarity'] < 0:
        val = -1
      else:
        val = 0
      return val

    df_en['polarity'] = df_en.apply(getPolarity2, axis=1)
    precov_en['polarity'] = precov_en.apply(getPolarity2, axis=1)
    precov_en['sentiment'] = precov_en.apply(getSentiment, axis=1)
    precov_en['polarity'] = precov_en.apply(normalizePolarity, axis=1)
    poscov_en['polarity'] = poscov_en.apply(getPolarity2, axis=1)
    poscov_en['sentiment'] = poscov_en.apply(getSentiment, axis=1)
    poscov_en['polarity'] = poscov_en.apply(normalizePolarity, axis=1)

    colors = ['red','grey','green'] 

    fig = make_subplots(rows=1,cols=6,subplot_titles=("Total","Events","Health","Research","Education","Image"))
    df_en['topic'] = df_en['topic'].astype(int)

    temp1 = precov_en.groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp2 = precov_en[precov_en['topic'] == 0].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp3 = precov_en[precov_en['topic'] == 1].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp4 = precov_en[precov_en['topic'] == 2].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp5 = precov_en[precov_en['topic'] == 3].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp6 = precov_en[precov_en['topic'] == 4].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()

    fig.add_trace(go.Bar(x=temp1['sentiment'], y=temp1['count']/temp1['count'].sum(), marker_color=colors), row=1, col=1)
    fig.add_trace(go.Bar(x=temp2['sentiment'], y=temp2['count']/temp2['count'].sum(), marker_color=colors), row=1, col=2)
    fig.add_trace(go.Bar(x=temp3['sentiment'], y=temp3['count']/temp3['count'].sum(), marker_color=colors), row=1, col=3)
    fig.add_trace(go.Bar(x=temp4['sentiment'], y=temp4['count']/temp4['count'].sum(), marker_color=colors), row=1, col=4)
    fig.add_trace(go.Bar(x=temp5['sentiment'], y=temp5['count']/temp5['count'].sum(), marker_color=colors), row=1, col=5)
    fig.add_trace(go.Bar(x=temp6['sentiment'], y=temp6['count']/temp6['count'].sum(), marker_color=colors), row=1, col=6)
    fig.update_layout(showlegend=False)

    fig.update_layout( title={'text': '','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top', 'font':{'size': 18}},font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black')
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey',tickformat="%")

    for i in range(1,7):
      fig['layout']['yaxis'+str(i)].update(title='',range=[0,0.9], autorange=False, )

    fig.update_annotations(font_size=20)
    fig.show()

    colors = ['red','grey','green'] 

    fig = make_subplots(rows=1,cols=6,subplot_titles=("Total","Events","Health","Research","Education","Image"))
    df_en['topic'] = df_en['topic'].astype(int)

    temp1 = poscov_en.groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp2 = poscov_en[poscov_en['topic'] == 0].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp3 = poscov_en[poscov_en['topic'] == 1].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp4 = poscov_en[poscov_en['topic'] == 2].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp5 = poscov_en[poscov_en['topic'] == 3].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()
    temp6 = poscov_en[poscov_en['topic'] == 4].groupby(['sentiment'])['created_at'].agg(['count']).reset_index()

    fig.add_trace(go.Bar(x=temp1['sentiment'], y=temp1['count']/temp1['count'].sum(), marker_color=colors), row=1, col=1)
    fig.add_trace(go.Bar(x=temp2['sentiment'], y=temp2['count']/temp2['count'].sum(), marker_color=colors), row=1, col=2)
    fig.add_trace(go.Bar(x=temp3['sentiment'], y=temp3['count']/temp3['count'].sum(), marker_color=colors), row=1, col=3)
    fig.add_trace(go.Bar(x=temp4['sentiment'], y=temp4['count']/temp4['count'].sum(), marker_color=colors), row=1, col=4)
    fig.add_trace(go.Bar(x=temp5['sentiment'], y=temp5['count']/temp5['count'].sum(), marker_color=colors), row=1, col=5)
    fig.add_trace(go.Bar(x=temp6['sentiment'], y=temp6['count']/temp6['count'].sum(), marker_color=colors), row=1, col=6)
    fig.update_layout(showlegend=False)

    fig.update_layout( title={'text': '','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top', 'font':{'size': 18}},font=dict(family="Times New Roman",size=15,color="black"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black')
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey',tickformat="%")

    for i in range(1,7):
      fig['layout']['yaxis'+str(i)].update(title='',range=[0,0.9], autorange=False, )

    fig.update_annotations(font_size=20)
    fig.show()

    temp = precov_en.groupby('user.name').size().reset_index(name='n_total')
    temp['positives'] = precov_en.groupby('user.name')['sentiment'].apply(lambda x: (x=='Positive').sum()).reset_index(name='count')['count']/temp['n_total']
    temp['neutrals'] = precov_en.groupby('user.name')['sentiment'].apply(lambda x: (x=='Neutral').sum()).reset_index(name='count')['count']/temp['n_total']
    temp['negatives'] = precov_en.groupby('user.name')['sentiment'].apply(lambda x: (x=='Negative').sum()).reset_index(name='count')['count']/temp['n_total']
    temp['positives'] = pd.Series(["{0:.2f}%".format(val * 100) for val in temp['positives'].round(4)], index = temp.index)
    temp['neutrals'] = pd.Series(["{0:.2f}%".format(val * 100) for val in temp['neutrals'].round(4)], index = temp.index)
    temp['negatives'] = pd.Series(["{0:.2f}%".format(val * 100) for val in temp['negatives'].round(4)], index = temp.index)

    temp2 = poscov_en.groupby('user.name').size().reset_index(name='n_total')
    temp2['covid-19 positives'] = poscov_en.groupby('user.name')['sentiment'].apply(lambda x: (x=='Positive').sum()).reset_index(name='count')['count']/temp2['n_total']
    temp2['covid-19 neutrals'] = poscov_en.groupby('user.name')['sentiment'].apply(lambda x: (x=='Neutral').sum()).reset_index(name='count')['count']/temp2['n_total']
    temp2['covid-19 negatives'] = poscov_en.groupby('user.name')['sentiment'].apply(lambda x: (x=='Negative').sum()).reset_index(name='count')['count']/temp2['n_total']
    temp2['covid-19 positives'] = pd.Series(["{0:.2f}%".format(val * 100) for val in temp2['covid-19 positives'].round(4)], index = temp2.index)
    temp2['covid-19 neutrals'] = pd.Series(["{0:.2f}%".format(val * 100) for val in temp2['covid-19 neutrals'].round(4)], index = temp2.index)
    temp2['covid-19 negatives'] = pd.Series(["{0:.2f}%".format(val * 100) for val in temp2['covid-19 negatives'].round(4)], index = temp2.index)
    temp2 = temp2.rename(columns={'n_total': 'covid-19 n_total'})

    temp = pd.merge(temp, temp2, on='user.name', how='outer')
    temp

    #@title
    import plotly.graph_objects as go
    names = ["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"]

    cal1 = temp[temp['user.name'] == "Caltech"]['positives'][0][:-1]
    cal2 = temp[temp['user.name'] == "Caltech"]['neutrals'][0][:-1]
    cal3 = temp[temp['user.name'] == "Caltech"]['negatives'][0][:-1]
    cu1 = temp[temp['user.name'] == "Cambridge University"]['positives'][1][:-1]
    cu2 = temp[temp['user.name'] == "Cambridge University"]['neutrals'][1][:-1]
    cu3 = temp[temp['user.name'] == "Cambridge University"]['negatives'][1][:-1]
    et1 = temp[temp['user.name'] == "ETH Zurich"]['positives'][2][:-1]
    et2 = temp[temp['user.name'] == "ETH Zurich"]['neutrals'][2][:-1]
    et3 = temp[temp['user.name'] == "ETH Zurich"]['negatives'][2][:-1]
    h1 = temp[temp['user.name'] == "Harvard University"]['positives'][3][:-1]
    h2 = temp[temp['user.name'] == "Harvard University"]['neutrals'][3][:-1]
    h3 = temp[temp['user.name'] == "Harvard University"]['negatives'][3][:-1]
    i1 = temp[temp['user.name'] == "Imperial College"]['positives'][4][:-1]
    i2 = temp[temp['user.name'] == "Imperial College"]['neutrals'][4][:-1]
    i3 = temp[temp['user.name'] == "Imperial College"]['negatives'][4][:-1]
    jhu1 = temp[temp['user.name'] == "Johns Hopkins University"]['positives'][5][:-1]
    jhu2 = temp[temp['user.name'] == "Johns Hopkins University"]['neutrals'][5][:-1]
    jhu3 = temp[temp['user.name'] == "Johns Hopkins University"]['negatives'][5][:-1]
    mit1 = temp[temp['user.name'] == "Massachusetts Institute of Technology (MIT)"]['positives'][6][:-1]
    mit2 = temp[temp['user.name'] == "Massachusetts Institute of Technology (MIT)"]['neutrals'][6][:-1]
    mit3 = temp[temp['user.name'] == "Massachusetts Institute of Technology (MIT)"]['negatives'][6][:-1]
    s1 = temp[temp['user.name'] == "Stanford University"]['positives'][7][:-1]
    s2 = temp[temp['user.name'] == "Stanford University"]['neutrals'][7][:-1]
    s3 = temp[temp['user.name'] == "Stanford University"]['negatives'][7][:-1]
    chi1 = temp[temp['user.name'] == "The University of Chicago"]['positives'][8][:-1]
    chi2 = temp[temp['user.name'] == "The University of Chicago"]['neutrals'][8][:-1]
    chi3 = temp[temp['user.name'] == "The University of Chicago"]['negatives'][8][:-1]
    ucb1 = temp[temp['user.name'] == "UC Berkeley"]['positives'][9][:-1]
    ucb2 = temp[temp['user.name'] == "UC Berkeley"]['neutrals'][9][:-1]
    ucb3 = temp[temp['user.name'] == "UC Berkeley"]['negatives'][9][:-1]
    ucl1 = temp[temp['user.name'] == "UCL"]['positives'][10][:-1]
    ucl2 = temp[temp['user.name'] == "UCL"]['neutrals'][10][:-1]
    ucl3 = temp[temp['user.name'] == "UCL"]['negatives'][10][:-1]
    uo1 = temp[temp['user.name'] == "University of Oxford"]['positives'][11][:-1]
    uo2 = temp[temp['user.name'] == "University of Oxford"]['neutrals'][11][:-1]
    uo3 = temp[temp['user.name'] == "University of Oxford"]['negatives'][11][:-1]


    fig = go.Figure(data=[
        go.Bar(marker=go.bar.Marker(color='rgb(50, 168, 82)'),name='Positive', x=names, y=[float(cal1),float(cu1),float(et1),float(h1),float(i1),float(jhu1),float(mit1),float(s1),float(chi1),float(ucb1),float(ucl1),float(uo1)],),
        go.Bar(marker=go.bar.Marker(color='rgb(184, 182, 182)'),name='Neutral', x=names, y=[float(cal2),float(cu2),float(et2),float(h2),float(i2),float(jhu2),float(mit2),float(s2),float(chi2),float(ucb2),float(ucl2),float(uo2)]),
        go.Bar(marker=go.bar.Marker(color='rgb(235, 64, 52)'),name='Negative', x=names, y=[float(cal3),float(cu3),float(et3),float(h3),float(i3),float(jhu3),float(mit3),float(s3),float(chi3),float(ucb3),float(ucl3),float(uo3)]),
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack',title="",legend={'traceorder':'normal'},paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    #fig.update_yaxes(visible=True, linecolor='black',tickformat="3%")
    fig.show()

    #@title
    import plotly.graph_objects as go
    names = ["Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"]

    cal1 = temp[temp['user.name'] == "Caltech"]['covid-19 positives'][0][:-1]
    cal2 = temp[temp['user.name'] == "Caltech"]['covid-19 neutrals'][0][:-1]
    cal3 = temp[temp['user.name'] == "Caltech"]['covid-19 negatives'][0][:-1]
    cu1 = temp[temp['user.name'] == "Cambridge University"]['covid-19 positives'][1][:-1]
    cu2 = temp[temp['user.name'] == "Cambridge University"]['covid-19 neutrals'][1][:-1]
    cu3 = temp[temp['user.name'] == "Cambridge University"]['covid-19 negatives'][1][:-1]
    et1 = temp[temp['user.name'] == "ETH Zurich"]['covid-19 positives'][2][:-1]
    et2 = temp[temp['user.name'] == "ETH Zurich"]['covid-19 neutrals'][2][:-1]
    et3 = temp[temp['user.name'] == "ETH Zurich"]['covid-19 negatives'][2][:-1]
    h1 = temp[temp['user.name'] == "Harvard University"]['covid-19 positives'][3][:-1]
    h2 = temp[temp['user.name'] == "Harvard University"]['covid-19 neutrals'][3][:-1]
    h3 = temp[temp['user.name'] == "Harvard University"]['covid-19 negatives'][3][:-1]
    i1 = temp[temp['user.name'] == "Imperial College"]['covid-19 positives'][4][:-1]
    i2 = temp[temp['user.name'] == "Imperial College"]['covid-19 neutrals'][4][:-1]
    i3 = temp[temp['user.name'] == "Imperial College"]['covid-19 negatives'][4][:-1]
    jhu1 = temp[temp['user.name'] == "Johns Hopkins University"]['covid-19 positives'][5][:-1] 
    jhu2 = temp[temp['user.name'] == "Johns Hopkins University"]['covid-19 neutrals'][5][:-1]
    jhu3 = temp[temp['user.name'] == "Johns Hopkins University"]['covid-19 negatives'][5][:-1]
    mit1 = temp[temp['user.name'] == "Massachusetts Institute of Technology (MIT)"]['covid-19 positives'][6][:-1]
    mit2 = temp[temp['user.name'] == "Massachusetts Institute of Technology (MIT)"]['covid-19 neutrals'][6][:-1]
    mit3 = temp[temp['user.name'] == "Massachusetts Institute of Technology (MIT)"]['covid-19 negatives'][6][:-1]
    s1 = temp[temp['user.name'] == "Stanford University"]['covid-19 positives'][7][:-1]
    s2 = temp[temp['user.name'] == "Stanford University"]['covid-19 neutrals'][7][:-1]
    s3 = temp[temp['user.name'] == "Stanford University"]['covid-19 negatives'][7][:-1]
    chi1 = temp[temp['user.name'] == "The University of Chicago"]['covid-19 positives'][8][:-1]
    chi2 = temp[temp['user.name'] == "The University of Chicago"]['covid-19 neutrals'][8][:-1]
    chi3 = temp[temp['user.name'] == "The University of Chicago"]['covid-19 negatives'][8][:-1]
    ucb1 = temp[temp['user.name'] == "UC Berkeley"]['covid-19 positives'][9][:-1]
    ucb2 = temp[temp['user.name'] == "UC Berkeley"]['covid-19 neutrals'][9][:-1]
    ucb3 = temp[temp['user.name'] == "UC Berkeley"]['covid-19 negatives'][9][:-1]
    ucl1 = temp[temp['user.name'] == "UCL"]['covid-19 positives'][10][:-1]
    ucl2 = temp[temp['user.name'] == "UCL"]['covid-19 neutrals'][10][:-1]
    ucl3 = temp[temp['user.name'] == "UCL"]['covid-19 negatives'][10][:-1]
    uo1 = temp[temp['user.name'] == "University of Oxford"]['covid-19 positives'][11][:-1]
    uo2 = temp[temp['user.name'] == "University of Oxford"]['covid-19 neutrals'][11][:-1]
    uo3 = temp[temp['user.name'] == "University of Oxford"]['covid-19 negatives'][11][:-1]


    fig = go.Figure(data=[
        go.Bar(marker=go.bar.Marker(color='rgb(50, 168, 82)'),name='Positive', x=names, y=[float(cal1),float(cu1),float(et1),float(h1),float(i1),float(jhu1),float(mit1),float(s1),float(chi1),float(ucb1),float(ucl1),float(uo1)]),
        go.Bar(marker=go.bar.Marker(color='rgb(184, 182, 182)'),name='Neutral', x=names, y=[float(cal2),float(cu2),float(et2),float(h2),float(i2),float(jhu2),float(mit2),float(s2),float(chi2),float(ucb2),float(ucl2),float(uo2)]),
        go.Bar(marker=go.bar.Marker(color='rgb(235, 64, 52)'),name='Negative', x=names, y=[float(cal3),float(cu3),float(et3),float(h3),float(i3),float(jhu3),float(mit3),float(s3),float(chi3),float(ucb3),float(ucl3),float(uo3)]),
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack',title="",legend={'traceorder':'normal'},paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.show()

    #@title
    import plotly.express as px

    df_en['user.name'] = df_en['user.name'].replace(['Massachusetts Institute of Technology (MIT)'],'MIT')

    lista = []
    for string in sorted(df_en['user.name'].unique()):
     temp = df_en[df_en['user.name'] == string ]
     temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['favorite_count'].agg(['mean']).reset_index()
     temp['user.name'] = string
     lista.append(temp)

    result = pd.concat(lista)

    fig = px.line(result, x="created_at", y="mean", facet_col="user.name",facet_col_wrap=4,labels={"mean": "","created_at":""},facet_row_spacing=0.08,facet_col_spacing=0.08)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(font=dict(size=10),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black',showticklabels=False)
    fig.update_yaxes(visible=True, linecolor='black',showticklabels=False,gridcolor='grey')

    #fig.add_vline(x="2020-03-11", line_width=3, line_dash="dash", line_color="red")

    rs=3
    cs=1

    for string in sorted(df_en['user.name'].unique()):
      if cs > 4:
        rs -= 1
        cs = 1
      if rs==3 and cs==1:
        flag = True
      else:
        flag = False
        
      temp = df_en[df_en['user.name'] == string ]
      temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['favorite_count'].agg(['mean']).reset_index()
      temp['user.name'] = string
      lista.append(temp)
      trace1 = go.Scatter(x=temp["created_at"], y=temp['mean'], line_color="#093426",showlegend=flag,name="favorites",mode='lines')
      fig.add_trace(trace1, row=rs, col=cs, exclude_empty_subplots=True)

      temp = df_en[df_en['user.name'] == string ]
      temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['retweet_count'].agg(['mean']).reset_index()
      temp['user.name'] = string
      lista.append(temp)
      trace2 = go.Scatter(x=temp["created_at"], y=temp['mean'], line_color="#7FBD32",showlegend=flag,name="retweets",mode='lines')
      fig.add_trace(trace2, row=rs, col=cs, exclude_empty_subplots=True)

      cs+=1

    fig.update_layout(font=dict(size=20))
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')
    fig.show()

    #@title
    import plotly.express as px

    df_en['user.name'] = df_en['user.name'].replace(['Massachusetts Institute of Technology (MIT)'],'MIT')

    lista = []
    for string in sorted(df_en['user.name'].unique()):
     temp = df_en[df_en['user.name'] == string ]
     temp = temp.groupby(pd.Grouper(key='created_at',freq='M'))['created_at'].agg(['count']).reset_index()
     temp['user.name'] = string
     lista.append(temp)

    result = pd.concat(lista)

    fig = px.line(result, x="created_at", y="count", facet_col="user.name",facet_col_wrap=4,labels={"count": "","created_at":""},facet_row_spacing=0.08,facet_col_spacing=0.08)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(font=dict(size=10),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black',showticklabels=False)
    fig.update_yaxes(visible=True, linecolor='black',showticklabels=False,gridcolor='grey')

    #fig.add_vline(x="2020-03-11", line_width=3, line_dash="dash", line_color="red")
    #fig.write_image("fig1.eps")
    fig.update_layout(font=dict(size=20))
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='grey')

    for i in range(1,13):
      fig['layout']['yaxis'+str(i)].update(title='', range=[0,500], autorange=False)
    fig.show()

    hei_stats = df_en.groupby(['user.name'])['created_at'].agg(['count','min', 'max']).reset_index()
    hei_stats.sort_values('count', ascending=False, ignore_index=True)
    hei_stats['min'] = hei_stats['min'].dt.strftime('%Y-%m-%d')
    hei_stats['max'] = hei_stats['max'].dt.strftime('%Y-%m-%d')
    hei_stats['avg day'] = hei_stats['count']/365
    hei_stats

    temp = df_en.groupby(['user.name'])
    temp.describe()


    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=np.array(df_en["retweet_count"]), y=np.array(df_en['polarity']),mode="markers"),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.array(df_en["favorite_count"]), y=np.array(df_en['polarity']),mode="markers"),row=1,col=2)
    fig.update_layout(font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',showlegend=False)
    fig.update_xaxes(visible=True, linecolor='black',title_text="Retweets", type="log", row=1, col=1)
    fig.update_xaxes(visible=True, linecolor='black',title_text="Favorites", type="log", row=1, col=2)
    fig.update_yaxes(visible=True, linecolor='black',title_text="Polarity", row=1, col=1)
    fig.show()

    #@title
    import plotly.figure_factory as ff
    import numpy as np
    import seaborn as sns

    fig = make_subplots(rows=3,cols=4,subplot_titles=("Caltech", "Cambridge University", "ETH Zurich", "Harvard University","Imperial College", "Johns Hopkins University", "MIT", "Stanford University","The University of Chicago","UC Berkeley","UCL","University of Oxford"))
    colors = list(sns.husl_palette(12, h=.5))

    cs = 1
    rs = 1
    i = 0
    for string in sorted(df_en['user.name'].unique()):
      if cs > 4:
        rs += 1
        cs = 1

      hist_data = []
      hist_data.append(df_en[df_en['user.name'] == string]["polarity"].tolist())
      group_label = [string]
      temp_fig = ff.create_distplot(hist_data, group_label,bin_size=.05)
      fig.add_trace(go.Histogram(temp_fig['data'][0],marker_color="rgb"+str(colors[i])), row=rs, col=cs)
      fig.add_trace(go.Scatter(temp_fig['data'][1],line=dict(color="rgb"+str(colors[i]), width=2)), row=rs, col=cs)
      cs += 1
      i += 1
    fig.update_layout(showlegend=False,height=600,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black')
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='white')

    for i in range(1,13):
      fig['layout']['xaxis'+str(i)].update(title='', range=[-1,1], autorange=False)

    fig.show()

    #@title
    import plotly.figure_factory as ff
    import numpy as np
    import seaborn as sns
    import matplotlib.colors as mcolors

    fig = make_subplots(rows=1,cols=5,subplot_titles=("Events","Health","Research","Education","Image"))
    colors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    topics = ["Events","Health","Research","Education","Image"]

    cs = 1
    for j in range(0,5):
      hist_data = []
      hist_data.append(df_en[df_en['topic'] == j]["polarity"].tolist())
      group_label = [topics[j]]
      temp_fig = ff.create_distplot(hist_data, group_label,bin_size=.05)
      fig.add_trace(go.Histogram(temp_fig['data'][0],marker_color=colors[j]), row=1, col=cs)
      fig.add_trace(go.Scatter(temp_fig['data'][1],line=dict(color=colors[j], width=2)), row=1, col=cs)
      cs += 1
      i += 1
    fig.update_layout(showlegend=False,height=600,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(visible=True, linecolor='black')
    fig.update_yaxes(visible=True, linecolor='black', gridcolor='white')

    for i in range(1,6):
      fig['layout']['xaxis'+str(i)].update(title='', range=[-1,1], autorange=False)

    fig.show()
    
    df_en.to_csv('df_en.csv', index=False)