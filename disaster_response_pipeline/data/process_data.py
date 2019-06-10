import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """loads the data from given paths and returns it in a pandas.DataFrame
    
    Arguments:
        messages_filepath {string} -- path to the message data
        categories_filepath {string} -- path to the categories data
    
    Returns:
        pandas.DataFrame -- a dataframe holding both messages and their categories, merged by using their id
    """
    messages = pd.read_csv(messages_filepath, engine='python')
    categories = pd.read_csv(categories_filepath, engine='python')
    return messages.merge(categories, on='id')


def clean_data(df):
    """Cleans dataframe holding disaster messages from duplicates, splits categories column and updates column labels
    
    Arguments:
        df {pandas.DataFrame} -- dataframe holding disaster response messages and their categories
    
    Returns:
        pandas.DataFrame -- dataframe with updated column labels, no duplicates and category columns
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # update column labels
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    # split categories into columns
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
        # let's make sure this stays binary; 1 means category applies; 0 (formerly 0 or 2) means it doesn't
        categories[column] = categories[column].apply(lambda x: 1 if x == 1 else 0)
    df = pd.concat([df.drop(columns='categories'), categories], axis=1)

    # remove duplicates
    print('[INFO] - {} duplicate ids found. Removing them now...'.format(df[df.duplicated(subset='id')].shape[0]))
    df.drop_duplicates(subset='id', inplace=True)

    return df


def save_data(df, database_filename, table_name='disaster_messages'):
    """Saves dataframe to given sqlite database
    
    Arguments:
        df {pandas.DataFrame} -- a pandas dataframe
        database_filename {string} -- name of a sqlite database the dataframe should be saved to
        table_name {string} -- name of the table the dataframe should be saved as; default: 'disaster_messages'
    
    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False)

    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('[INFO] - Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('[INFO] - Cleaning data...')
        df = clean_data(df)
        
        print('[INFO] - Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('[INFO] - Cleaned data saved to database!')
    
    else:
        print('[WARNING] - Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()