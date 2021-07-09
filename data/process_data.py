# import libraries
import pandas as pd
import sqlalchemy as sa
import sys


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages, categories


def clean_data(messages, categories):
    categories_fields = categories['categories'].str.split(';',expand=True)
    row = categories_fields.head(1)
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x.str[:-2],axis=1).values[0]
    categories_fields.columns = category_colnames
    
    for column in categories_fields:
        # set each value to be the last character of the string
        categories_fields[column] = categories_fields[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories_fields[column] = categories_fields[column].astype(int)
    
    categories.drop(columns='categories',inplace=True)
    categories = pd.concat([categories, categories_fields],axis=1)

    df = messages.merge(categories,on=['id'])
    
    df.drop_duplicates(subset=['id'],keep='first',inplace=True)    
    df.dropna(subset=['id'],inplace=True)

        
    return df


def save_data(df, database_filename):
    
    engine = sa.create_engine('sqlite:///data/testdb')
   # df.to_sql('messages', engine, index=False)
    df.to_csv('data/response.csv',index=False)

    pass  




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()