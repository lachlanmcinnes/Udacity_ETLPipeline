import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    '''
        Create a dataframe from the sql database where the cleaned data is located
        
        INPUT:
            location string of SQL Database
        
        OUTPUT:
            pandas dataframe
    '''
    
    #Create engine to connect to SQL Database
    engine = create_engine(f'sqlite:///{database_filepath}')
    #Create Dataframe
    df = pd.read_sql("SELECT * FROM cleaned_ETL_data", engine)
    
    X = df['message']
    Y = df.drop(columns=['message','genre','id','original'])
    
    return X, Y, Y.columns


def tokenize(text):
    '''
        Build a Tokenization function which lemmatize the words, converts to lower case and strips any white space
        
        INPUT:
            string text
        
        OUTPUT:
            Cleaned text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
        Create a PipeLine and GridSearchCV to train and predict incoming data
        
        INPUT: 
            None
        
        OUTPUT:
            Machine Learning Model
    '''
    
    #Create Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Report the f1 score, precision and recall for each output category of the dataset
        
        INPUT:
            machine learning model
            X_test data
            Y_test data
            Categories to evaluate
    '''
    
    #predict based on X_test data
    y_pred = model.predict(X_test)
    
    #Print the evaluation outcomes
    target_names = ['Val = 0', 'Val = 1']
    for i,col in enumerate(category_names):
        rep = classification_report(Y_test[col],y_pred[:,i], target_names=target_names)
        print(f'report for {col}')
        print(rep)
        print('\n')


def save_model(model, model_filepath):
    '''
        Save trained model as a pickle file
        
        INPUT: 
            trained machine learning model
            
        OUTPUT:
            pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()