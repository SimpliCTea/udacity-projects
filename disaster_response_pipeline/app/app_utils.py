import re
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

def tokenize(text):
    """Takes text and normalizes, tokenizes, strips and lemmatizes it and removes stopwords.
    
    Arguments:
        text {string} -- text to be tokenized
    
    Returns:
        list -- array of word tokens based on the given text string
    """
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    token_list = word_tokenize(text)
    token_list = [token.strip() for token in token_list if token not in stopwords.words("english")]
    token_list = [lemmatizer.lemmatize(token) for token in token_list]
        
    return token_list

def get_category_counts():
    """Creates and returns the counts (occurence) of all categories
    
    Returns:
        dict -- labels: list with category names, counts: pandas.Series with the counts of the categories
    """
    category_df = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_counts = {
        'labels': category_df.columns,
        'counts': category_df.sum()
    }
    return category_counts

def create_topwords_img(top_k=100):
    """Creates a wordcloud with the top k words (by occurence in dataset) and saves it as a png
    
    Keyword Arguments:
        top_k {int} -- top k words to be displayed in the word cloud (default: {100})
    
    Returns:
        This function does not return a value.
    """
    print('[appUtils]: Creating word cloud with top {} words...'.format(top_k))
    vectorizer = CountVectorizer(tokenizer=tokenize)
    doc_matrix = vectorizer.fit_transform(df['message'])
    bag_of_words = pd.DataFrame(doc_matrix.toarray(), columns=vectorizer.get_feature_names())
    most_common_words = bag_of_words.sum().nlargest(top_k)
    
    wordcloud = WordCloud(background_color="white", width=1920, height=1080)
    wordcloud.fit_words(most_common_words)
    wordcloud.to_file('./static/topwords.png')
    print('[appUtils]: Word cloud created. Saved under app/static/topwords.png')