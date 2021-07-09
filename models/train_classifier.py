import numpy as np
import pandas as pd
import sklearn as sk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import nltk
import re
import sqlalchemy as sa
import pickle

import string 
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

    
def model_pipeline(clf_type):
    
    if clf_type == 'mlp':
        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(50,), random_state=1)
        
    elif clf_type == 'nb' : 
        model = MultinomialNB()
        
        
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(model, n_jobs=-1))
    ])
    
    return pipeline


    
def load_data(database_filepath):
    
    engine = sa.create_engine('sqlite:///'+database_filepath)
    #print(database_filepath)
    print(engine)
    df = pd.read_sql_table('messages',con=engine)
    category_names = df.columns.values.tolist()[4:]
   
    df['original'].fillna('',inplace=True)
    df.dropna(inplace=True)

    X = df.iloc[:,1]
    y = df.iloc[:,4:].values
    y = y.astype(int)

    return X,y, category_names


def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words: 
            clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    model = model_pipeline('mlp')
    #model.fit(X_train, y_train)
   

    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    #confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == Y_test).mean()
    print("Labels:", category_names)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    
    precision = dict()
    recall = dict()
    average_precision = dict()

    y_pred[y_pred==2]=0
    Y_test[Y_test==2]=0

  
    for i in range(len(category_names)):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_pred[:, i])
        precision_avg = average_precision[i]
        recall_avg = recall[i].mean()
        F1_score = 2*(recall_avg * precision_avg) / (recall_avg + precision_avg)
        accuracy = (y_pred[:,i] == Y_test[:,i]).mean()
        print (category_names[i], ':\n\t Accuracy : ', accuracy, ':\n\t Precision : ', precision_avg,':\n\t Recall : ',recall_avg, ':\n\t F1 Score : ',F1_score)

    return

def save_model(model, model_filepath):
    filename = 'classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))
    

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