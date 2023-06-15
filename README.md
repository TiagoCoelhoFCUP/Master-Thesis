# Master-Thesis
## Higher Education Institutions Social Media Content Strategies

In this dissertation we create an automatic
system capable of identifying Higher Education Institutions Twitter communication strategies and the topics that emerge from their publication contents. Furthermore, with the current Covid-19 pandemic causing major consequences in many different fields (political, economic, social, educational) beyond the spread of the disease itself, we determine the impact of the pandemic on the identified HEI content strategies. We then gage the predictive capability of the obtained editorial models by attempting to predict the engagement and the dominant topic of a publication. We gathered and analyzed more than 18k Twitter publications from 12 of the top HEI according to the 2019 Center for World University Rankings (CWUR). Utilizing machine learning techniques, and topic modeling, we determined the emergent content topics across all HEI: Education, Faculty, Employment, Research, Health and Society. Then, we characterized the editorial strategy of each HEI before, and during, the Covid-19 pandemic and were able to observe significant differences between the specified periods. Lastly, with respect to the predictive capability of the models, we determine that the information gathered is not enough to be able to accomplish the proposed tasks

<div align="center">
  <img src="https://github.com/TiagoCoelhoFCUP/Master-Thesis/assets/13381706/17b05d60-e5b4-4276-b298-7cac9928e6eb" alt="Poster">
</div>

The work developed in this dissertation was the subject of a scientific paper accepted by the
21st International Conference on Computer Sciences and their Applications (ICCSA)
with the name 'Covid-19 Impact on Higher Education Institution's Social Media Content Strategy', and
consequently included in Springer Lecture Notes in Computer Science (DOI: 10.1007/978-
3-030-86960-1_49) as well as indexed by Scopus, EI Engineering Index, Thomson Reuters
Conference Proceedings Citation Index (included in ISI Web of Science), and several other
indexing services.
This work was also the subject of an article accepted in the IEEE BigData 2021 conference titled "Analysis of Top-Ranked HEI Publications’ Strategy
on Twitter".

```
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
```




