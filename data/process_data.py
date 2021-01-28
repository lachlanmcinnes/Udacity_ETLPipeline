import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        Takes data from two csv files, splits the categories column into seperate columns and joins together based on the "id" attribute
    
        INPUT:
            messages csv file
            categories csv file
        
        OUTPUT:
            combined pandas dataframe
    '''
    
    #Load csv files into dataframes and merge
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    '''
        Take a dataframe and spearate categories column into individual data points.  Remove duplicates and return the concatenated dataframe
        
        INPUT:
            pandas dataframe
            
        OUTPUT:
            cleaned pandas dataframe
    
    '''
    
    #split the categories column into seperate columns
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = 0
    # use this row to extract a list of new column names for categories.
    category_colnames = list(categories.iloc[row].str[:-2])
    #apply column names to dategories dataframe
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #drop the categories column from the original dataframe
    df.drop(columns=['categories'], inplace=True)
    
    #concatenate the original dataframe with the new categories dataframe
    df_concat = pd.concat([df, categories], sort=False, axis=1)
    
    #remove duplicate rows
    df_dropped_duplicates = df_concat.drop_duplicates()
    
    return df_dropped_duplicates


def save_data(df, database_filename):
    '''
        Dump the contents of the pandas dataframe into a defined sql database location.  Table name will be "cleaned_ETL_data"
        
        INPUT:
            pandas dataframe
            database file name
        
        OUTPUT:
            None
    '''
    
    #Create Engine for sql database
    engine = create_engine(f'sqlite:///{database_filename}')
    #Dump contents of DataFrame to sql database
    df.to_sql('cleaned_ETL_data', engine, index=False)


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