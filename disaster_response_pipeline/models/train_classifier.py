import sys
import re
import joblib
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support


def load_data(database_filepath):
    """Loads the data from the given database and splits it into X and Y
    
    Arguments:
        database_filepath {string} -- path (including filename) to the database with the message data
    
    Returns:
        tuple -- X pandas.Series with the messages, Y pandas.DataFrame with the categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, Y


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
    token_list = [lemmatizer.lemmatize(t.strip()) for t in token_list if t not in stopwords.words("english")]
    return token_list


def build_model(with_gridsearch=False):
    """Creates a pipeline with vectorizer, tfidf and classifier. Classifier is initiated with 
    default values, optimized on the initial training set. For new data a new gridsearch can 
    be build with preset parameters.
    
    Keyword Arguments:
        with_gridsearch {bool} -- Setting this to true will place the pipeline in a 
        scikit-learn GridSearchCV (default: {False})
    
    Returns:
        [scikit-learn model] -- If with_gridsearch is False this will return a pipeline model, otherwise
        it will return a grid search model
    """
    model = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(
            max_features=0.1,
            min_samples_split=2,
            n_estimators=50
        )))],
        verbose=True
    )
    if with_gridsearch:
        print('[INFO] - Using grid search to optimize model...')
        parameters = {
            'classifier__estimator__n_estimators': [10, 50],
            'classifier__estimator__min_samples_split': [2,4,6],
            'classifier__estimator__max_features': [.1, .2, 'sqrt', 'log2']
        }
        #model = GridSearchCV(model, parameters, n_jobs=-1, verbose=3)
        model = GridSearchCV(model, parameters, verbose=3)
    return model


def evaluate_model(model, X_test, Y_test):
    """Takes a model and makes a prediction on a test set. The evaluation of the prediction is printed out.
    
    Arguments:
        model {scikit-learn estimator} -- a valid scikit-learn estimator with a predict method
        X_test {pandas.DataFrame} -- a dataframe with the test data
        Y_test {pandas.DataFrame} -- a dataframe with the expected results
    """
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean().mean()
    report = classification_report(Y_test, Y_pred, target_names=Y_test.columns)
    print('='*70)
    print('[INFO] - Classification Report:')
    print('-'*70)
    print(report)
    print('-'*70)
    print('[INFO] - Overall accuracy: {:.2f}%'.format(accuracy*100))
    print('='*30 + ' END REPORTS ' + '='*30)


def save_model(model, model_filepath):
    """Saves model as pickle file using python joblib
    
    Arguments:
        model {scikit learn estimator} -- the model to be saved
        model_filepath {string} -- the path including filename where the model is to be saved
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) in [3,4]:
        database_filepath, model_filepath = sys.argv[1:3]
        with_gridsearch = False
        if len(sys.argv) == 4 and sys.argv[-1] == 'with_gridsearch':
            with_gridsearch = True

        print('[INFO] - Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('[INFO] - Building model...')
        model = build_model(with_gridsearch)
        
        print('[INFO] - Training model...')
        model.fit(X_train, Y_train)
        
        print('[INFO] - Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('[INFO] - Saving model...\n    MODEL: {}'.format(model_filepath))
        if with_gridsearch:
            model = model.best_estimator_
        save_model(model, model_filepath)

        print('[INFO] - Trained model saved!')

    else:
        print('[WARNING] - Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. Optionally, you can '\
              'optimize the model with a gridsearch; to do so, provide with_gridsearch as '\
              'your third argument. \n\nExample: python train_classifier.py'\
              ' ../data/DisasterResponse.db classifier.pkl with_gridsearch')


if __name__ == '__main__':
    main()