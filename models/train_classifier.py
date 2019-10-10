import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    loads the data from given database
    Return: X, y, category_names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    return X, y, y.columns.tolist()
    


def tokenize(text):
    """
    Cleans and tokenize the text
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    tokens = [lemmatizer.lemmatize(tok, pos='v').strip() for tok in tokens]
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    tokens = [word for word in tokens if word not in stopwords_]

    return tokens
    


def build_model():
    """
    build a ML model based on Pipeline
    Return: the model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1,2)),
        'clf__estimator__n_estimators' : [50, 100]
        }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    return cv
        


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates the model and print it
    Args: 
        model: the scikit fitted model
        X_test: test set
        Y_test: test classifications
        category_names: as named
    Returns:
        None
    """
    
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print ('Accuracy:\n {}'.format(accuracy))
    try:
        # runs in jupyter lab
        print(classification_report(y_test, y_pred, target_names=category_names))
    except:
        for i in range(len(category_names)):
            print('{}:\n {}'.format(category_names[i],
                  classification_report(y_test.iloc[:, i], y_pred[:, i])))
        print('classification_report on whole dataset not working')
        pass


def save_model(model, model_filepath):
    """
    saves the model to given file
    Return: 
        None
    """
    
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