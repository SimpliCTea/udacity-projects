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
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, Y


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    token_list = word_tokenize(text)
    token_list = [token.strip() for token in token_list if token not in stopwords.words("english")]
    token_list = [lemmatizer.lemmatize(token) for token in token_list]
        
    return token_list


def build_model(with_gridsearch):
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
        parameters = {
            'classifier__estimator__n_estimators': [10, 50],
            'classifier__estimator__min_samples_split': [2,4,6],
            'classifier__estimator__max_features': [.1, .2, 'sqrt', 'log2']
        }
        model = GridSearchCV(model, parameters, n_jobs=-1, verbose=3)
    return model


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean().mean()

    print('='*50)
    print('[INFO] - Classification Reports:')
    print('-'*50)
    for i, column in enumerate(Y_test):
        label_report = classification_report(Y_test.values[i], Y_pred[i])
        print('Label: {}'.format(column))
        print('-'*50)
        print(label_report)
        print('\n')
    print('-'*50)
    print('[INFO] - Overall accuracy: {:.2f}%'.format(accuracy*100))
    print('='*20 + ' END REPORTS ' + '='*20)


def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) in [3,4]:
        database_filepath, model_filepath = sys.argv[1:]
        with_gridsearch = False
        if len(sys.argv) == 4:
            with_gridsearch = sys.argv[-1]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(with_gridsearch)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        if with_gridsearch:
            model = model.best_estimator_
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. Optionally you can '\
              'optimize the model with a gridsearch; to do so, provide True as'\
              'your third argument. \n\nExample: python train_classifier.py'\
              ' ../data/DisasterResponse.db classifier.pkl True')


if __name__ == '__main__':
    main()