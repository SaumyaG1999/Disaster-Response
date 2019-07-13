import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Load Data function
    
    Arguments:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Output:
        df -> Loaded dasa as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = "outer" , on = ["id"])
    return df


def clean_data(df):
    
    """
    Clean Data function
    
    Arguments:
        df -> raw data as Pandas DataFrame
    Outputs:
        df_new -> clean data as Pandas DataFrame
    """
    categories = df["categories"].str.split(";",expand = True)
    
    row = categories.iloc[0,:]
    category_colnames = row.str.split("-",expand = True)[0]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str.split("-").str[1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop("categories", axis = 1)
    df_new = pd.concat([df, categories] , axis = 1)
    
    df_new = df_new[df_new.duplicated() == False]
    return df_new


def save_data(df_new, database_filename):
    
    """
    Save Data function
    
    Arguments:
        df_new -> Clean Pandas DataFrame
        database_filename -> database file (.db) destination path
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df_new.to_sql('messages', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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