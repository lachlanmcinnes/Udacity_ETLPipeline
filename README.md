# Disaster Response Pipeline Project

This github repository is used to identify different categories diffeerent social media messages trigger, specifically around emergency needs.  By creating a machine learning model we hope to predict incoming message needs to helping emergency services provide aid where needed

### Motivations:
THrough different natural disasters people who have been impacted will require different things in order to stay safe.  For example, those impacted by floods may cut off supply routes, making it hard to recieve medical suplies or food.  Through this project we are hope to analyse each incoming message and see how the different categories are triggered, thus allowing emergency responders better utilise thir own resources.

### Files:
* python file - used to wrangle data
* python file - used to train model and export as pickle file
* python file - contains flask code to run website
* csv fils - containing the messages and categories

### Python Libraries
* pandas
* numpy
* flask
* sqlalchemy
* sklearn
* ntlk
* pickle

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
