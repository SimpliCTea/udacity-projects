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
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    token_list = word_tokenize(text)
    token_list = [token.strip() for token in token_list if token not in stopwords.words("english")]
    token_list = [lemmatizer.lemmatize(token) for token in token_list]
        
    return token_list

def get_class_counts():
    class_df = df.drop(columns=['id', 'message', 'original', 'genre'])
    class_counts = {
        'labels': class_df.columns,
        'counts': class_df.sum()
    }
    return class_counts

def create_topwords_img(n_words=100):
    print('[appUtils]: Creating word cloud...')
    vectorizer = CountVectorizer(tokenizer=tokenize)
    doc_matrix = vectorizer.fit_transform(df['message'])
    bag_of_words = pd.DataFrame(doc_matrix.toarray(), columns=vectorizer.get_feature_names())
    most_common_words = bag_of_words.sum().nlargest(n_words)
    
    wordcloud = WordCloud(background_color="white", width=1920, height=1080)
    wordcloud.fit_words(most_common_words)
    wordcloud.to_file('./static/topwords.png')
    print('[appUtils]: Word cloud created. Saved under app/static/topwords.png')
    return None