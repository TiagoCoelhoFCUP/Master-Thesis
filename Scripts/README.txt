Before Running
--------------
- Make sure you have Python version 3.7.4 installed.
- Make sure to write (overwriting the previous entry) the full path to the 'mallet-2.0.8' folder on your machine into the 'mallet_path.txt" file. 
- Run the following commands to install the required libraries in their correct versions: 
>>> pip install pandas
>>> pip install spacy==3.0.5
>>> pip install plotly==4.14.3
>>> pip install emoji==1.2.0
>>> pip install gensim==3.8.3
>>> pip install nltk
>>> pip install matplotlib
>>> python -m spacy download en_core_web_sm
>>> pip install psutil
>>> pip install wordcloud
>>> pip install seaborn
>>> pip install datetime
>>> pip install textblob
>>> pip install adjustText
>>> pip install sklearn

Run Order
----------
exp1_editorial -> exp2_editorial -> any predictive analysis script

(due to the way the notebooks were originally written, some sentiment analysis graphs are also obtained by running epx1_editorial)
(exp1_editorial generates the file df_en.csv, needed for exp2_editorial and predictive analysis)
(exp2_editorial generates the file df_en2.csv, needed for predictive analysis)

