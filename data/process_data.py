# Import required libraries
import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, how='outer', left_on='id', right_on='id')


def clean_data(df):
    # Now we first determen the column names
    columns = df['categories'].str.replace("-0","").str.replace("-1","").str.replace("-2","").drop_duplicates().str.split(';')[0]

    # And split the 'categories' column
    df[columns] = df['categories'].str.split(';',expand=True)

    # The columns are still an object/string. Now make it a number
    for column in columns :
        df[column]=df[column].str.replace(column+"-","").astype(int)

    # We don't need the column 'categories' anymore, so delete it
    df.drop('categories',1,inplace=True)
    
    # There are some duplicate records, so drop them
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        # Check if input files exists
        if not((os.path.exists(messages_filepath))&(os.path.exists(categories_filepath))) :
            if not(os.path.exists(messages_filepath)) :
                print('Can\'t find : %s' % messages_filepath)
            if not os.path.exists(categories_filepath) :
                print('Can\'t find : %s' % categories_filepath)
            return
        
        # Check if ouptut file already exists
        if (os.path.exists(database_filepath)) :
            print ("Output file %s already exists\nPlease delete file first if you want to update" % database_filepath)
            return
                  
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories\n'\
              'datasets as the first and second argument respectively, as\n'\
              'well as the filepath of the database to save the cleaned data\n'\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()