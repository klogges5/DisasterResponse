# Disaster Response Pipeline Project

## Project Description
In ths project Figure Eight has provided labeled tweet and text messages after a disaster.
The project is broken down in three steps:
1. Read the messages and categories from csv-file, clean the data and write it to a database
2. Read the data from database and build a ML-model via Pipeline and use of GridSearch. Save the model to a pickle file
3. Read the database and the pickle file to show some graphics and predict messages on a website.


### Libraries needed
- re
- Pandas
- SQLAlchemy
- Pickle
- NLTK
- SKLearn
- Flask
- Plotly


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/




