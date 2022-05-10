from bs4 import BeautifulSoup
import requests
import numpy as np
from time import sleep
from random import randint
import pandas as pd
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder



def next_page():
    Topic = []
    Authors = []
    PublishedDate = []
    Info = []
    pages = np.arange(1, 51, 1)
    for page in pages:
        page = requests.get('https://calhoun.nps.edu/handle/10945/16/discover?rpp=20&etal=0&group_by=none&page=' + str(page))
        soup = BeautifulSoup(page.text, 'html.parser')
        All_topics = soup.find_all('div', class_='row ds-artifact-item list-view-item')
        sleep(randint(2, 10))
        for topic in All_topics:
            Topic_ = topic.find('div', class_='col-sm-9 artifact-description').h4.text
            Topic.append(Topic_)
        
            try:
                Topic_Authors = topic.find('div', class_='artifact-info').span.text
                Authors.append(Topic_Authors)
            except:
                Authors.append('No author')
                continue
            try:
                Published_date = topic.find('span', class_='date').text
                PublishedDate.append(Published_date)
            except:
                PublishedDate.append('No date')
                continue
            try:
                Topic_description = topic.find('div', class_='abstract').text
                Info.append(Topic_description)
            except:
                Info.append('no description')
                continue
            print('done')
    print(len(Topic), len(Authors), len(PublishedDate), len(Info))
    col1 = pd.Series(list(Topic), name='Topics')
    col2  = pd.Series(list(Authors), name='Authors')
    col3 = pd.Series(list(PublishedDate), name='Date')
    col4 = pd.Series(list(Info), name='Description')
    df = pd.concat([col1,col2,col3,col4], axis=1)
    print(df.head(5))
    df.to_csv('TrialData1.csv')
          
def trial_analysis():
    description_list = []
    stopwords = []
    min_len = 3
    stopwords_file = open('stopwords_en.txt', 'r', encoding='utf8')
    df = pd.read_csv('TrialData1.csv')
    pd.set_option('display.max_columns', None)
    df = df.drop('Unnamed: 0', 1)
    df = df.dropna(subset=['Description'])
    for word in stopwords_file:
        stopwords.append(word.strip())
    stopwords.extend(['research','program','Unknown', 'author', 'thesis','description',
                      'project','method','conducted','using','analysis','used','analyze','study',
                      'naval','navy'])             
    Topic_keywords = df['Description'].astype(str)
    Authors_ = df['Authors'].astype(str)
    Author_list = []
    for line in Topic_keywords:
        parts = line.strip().split()
        for word in parts:
            word_l = word.lower().strip()
            if word_l.isalpha():  # removing non alpha characters
                if len(word_l) > min_len:  # removing words less than 3 characters
                    if word_l.lower() not in stopwords:  # removing words in stop word file
                        if word_l not in string.digits:  # removing numeric characters
                            if word_l not in string.punctuation:  # removing punctuations
                                description_list.append(word_l.lower())
    Clean_Topic = df['Topics']
    Clean_topic_text = []
    Author_list = []
    for line in Clean_Topic:
        parts = line.strip().split()
        for word in parts:
            if word.lower() not in stopwords:
                if word.lower() not in string.punctuation:
                    if word.isalpha():
                        if len(word) > min_len:
                            Clean_topic_text.append(word)
    for line in Authors_:
        parts = line.strip().split()
        for word in parts:
            if word.lower() not in stopwords:
                if word.isalpha():
                    Author_list.append(word)
    
    # Bi grams for Topics
    word_list_str = ' '.join(Clean_topic_text)
    word_list_str_tokenize = nltk.word_tokenize(word_list_str)  # tokenize the file for printing bi-grams
    Topic_bi_gram = list(nltk.bigrams(word_list_str_tokenize))  # printing the bi-grams using nltk
    print('\nThe first 50 Bi-grams for the Topics are\n' + str(Topic_bi_gram[:50])) # printing first 10
    print('There are total ' + str(len(Topic_bi_gram)) + ' Bi-grams for Topics')  # printing number of bi-grams
    
    # Bi grams for Authors
    Author_list_str = ' '.join(Author_list)
    Author_list_str_tokenize = nltk.word_tokenize(Author_list_str)  # tokenize the file for printing bi-grams
    Author_bi_gram = list(nltk.bigrams(Author_list_str_tokenize))  # printing the bi-grams using nltk
    print('\nThe first 20 Bi-grams for authors are\n' + str(Author_bi_gram[:20]))  # printing first 10
    print('\nThere are total ' + str(len(Author_bi_gram)) + ' Bi-grams for Authors')  # printing number of bi-grams
    Frequent_authors = nltk.FreqDist(Author_bi_gram).most_common(20)
    print('\nAuthors that worked together more frquently are:\n' + str(Frequent_authors))
    
    # Common Words in the description
    common_words_topic = nltk.FreqDist(description_list).most_common(30)
    print(common_words_topic)
    
    # Defining the wordcloud parameters
    wc = WordCloud(background_color="white", max_words=2000)
    wc1 = WordCloud(background_color="white", max_words=2000)

    # Generate word cloud
    all_words_string = ' '.join(Clean_topic_text)
    description_list_str = ' '.join(description_list)
    wc1.generate(description_list_str)
    wc.generate(all_words_string)
    

    # Show the cloud
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    plt.imshow(wc1)
    plt.axis('off')
    plt.show()

def author_collab():
    

    df = pd.read_csv('TrialData1.csv')
    pd.set_option('display.max_columns', None)
    df['Authors'] = df['Authors'].str.replace('[.;]','')
    data = list(df['Authors'].apply(lambda x:x.split(",")))
    a = TransactionEncoder()
    a_data = a.fit(data).transform(data)
    df1 = pd.DataFrame(a_data,columns=a.columns_)
    df1 = df1.replace(False,0)
    df1 = df1.replace(True,1)
    corr = df1.corr()
    # Pairs to drop
    pairs_to_drop = set()
    cols = df1.columns
    for i in range(0, df1.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    au_corr = corr.abs().unstack()
    labels_todrop = pairs_to_drop
    au_corr = au_corr.drop(labels=labels_todrop).sort_values(ascending=False)
    print(au_corr[0:5])
    corr.style.background_gradient(cmap='coolwarm')
    sns.heatmap(corr, vmin=0.99, vmax=1, cmap="YlGnBu")

    # Finding support level for unique authors 
    df1 = apriori(df1, min_support=0.005, use_colnames = True)
    print(df1)

