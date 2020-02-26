import sys
import pandas as pd
import os
from sqlalchemy import create_engine
import re

# Machine learn library
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Text processing libraties
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Import NLTK resources
import nltk
nltk.download([ 'stopwords'])

# Use Pickle to save model
import pickle

def load_data(database_filename):
    """ Load the SQLite datafile and reurn the X (messages), Y (categories) and an array of the categorie names
        
        Parameters
        ----------
        database_filename = the file location of the SQLite database
        
        Returns
        -------
        X, Y, category_names
    """     
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = pd.read_sql_table('messages', engine)
    
    Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 
        'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 
        'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 
        'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
        'other_weather', 'direct_report']]
    
    X = df['message']
    
    category_names = Y.columns
  
    return X, Y, category_names


def tokenize(text):
     
    # Replace URL's
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Replace special signs to text
    ReplaceList = [ ["&amp;", "&"], [ "&gt;", ">"], [ "&lt;", "<"], [ "&nbsp;" , " "], [ "&quot;" , "\""], 
                    ["£", " GBP "], ["$", " Dollar "], ["€", " Euro "], 
                    ["%", " percent "], ["&", " and "],    ["°", " degree "]
                  ]
    
    for item in ReplaceList :
        text = text.replace(item[0], item[1]) 
        
    # Punctuation Removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
      
    # Tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    # Remove stop words, words that are not important for a scentence
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens

def build_model():
    return Pipeline([
        ('vect', CountVectorizer( tokenizer=tokenize, 
                                  strip_accents='unicode',
                                )),           # Vectorize the text, using the tokenize function
        ('tfidf', TfidfTransformer()),                           # Make calculate the IDF values -> https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/
        ('clf', MultiOutputClassifier(RandomForestClassifier())) # Define a multioutput randomforest
    ])
     


def evaluate_model(model, X_test, Y_test, category_names):
    
    # predict on test data with tuned params
    y_pred = model.predict(X_test)
    
    # loop trough catagories
   
    for k in range(0, Y_test.shape[1] ) : 
        print ("\x1b[31m", end="") # Print catagory names in red, for readability
        print ( Y_test.columns[k] )
        print ("\x1b[00m", end="") # Print report in black
        print (classification_report(Y_test.iloc[:,k], y_pred[:,k])) 
    
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        if not os.path.isfile(database_filepath):
            print ( "File not found")
            return
                
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
