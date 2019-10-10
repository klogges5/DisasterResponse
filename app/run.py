import json
import re
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    # counts of related messages
    related = y[y.related == 1].sum()[1:].sort_values(ascending=False)
    
    # correlation on categories
    correlation_categories = y.corr().values
    category_names = list(y.columns)
    
    # count categories
    cat = y.sum().sort_values(ascending=False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=cat
                )
            ],

            'layout': {
                'title': 'Sums of Categories',
                'yaxis': {
                    'title': "Sum"
                },
                'xaxis': {
                    'tickangle': -45
                },
                'margin' : dict(
                        b = 200)
            }
        },
        {
            'data': [
                Bar(
                    x=related.index,
                    y=related
                )
            ],

            'layout': {
                'title': 'count of related disaster types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "disaster types",
                    'tickangle' : -45
                },
                'margin' : dict(
                        b = 200)
            }
        },
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names,
                    z=correlation_categories
                )    
            ],

            'layout': {
                'title': 'Correlation of Categories',
                'xaxis': {'tickangle': -45},
                'yaxis': {'automargin': True},
                'margin' : dict(
                        b = 200)
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(port=3001, debug=True)


if __name__ == '__main__':
    main()