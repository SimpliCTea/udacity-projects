# Disaster Response Pipeline Project

In this project a basic web-application was built that allows you to analyse a message and categorize it to quickly understand its main topics. The messages used for training the model are from a disaster message dataset provided by Figure Eight (https://www.figure-eight.com/dataset/combined-disaster-response-data/). You can however provide your own message dataset and retrain the model on it, even perform another gridsearch if you want to optimize the model with the new dataset.

## Requirements:

The projects is mostly scripted in python 3 with a bit of javascript for the web-application. For the pipelines I used pandas, scikit-learn, SQLAlchemy, joblib and nltk for text processing. The wordcloud is created using the wordcloud package (https://amueller.github.io/word_cloud/), which is also available on conda-forge. The web-applocication also makes use of Flask and plotly. 

A list with versions used for this project can be found in requirements.txt in the project's root directory. In case there are compatibility issues check here first, you may run a different version of one of the packages. 

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - The classifier is preset with the optimal parameters found with the initial training data. If you have new data and want to see whether you can further optimize it on the new data, you can perform a gridsearch by adding a fourth argument: with_gridsearch
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl with_gridsearch`

2. Run the following command in the app's directory to run your web app. 
    `python run.py`
    Doing this for the first time will take a moment as the app is first creating a wordcloud with the top 100 words in the dataset. Once the wordcloud has been created and saved as a PNG file the server will start running. If you have new data and want to recreate the wordcloud, go to delete the PNG at app/static/topwords.png and restart the server with the command mentioned above.

3. Go to http://0.0.0.0:3001/

## Troubleshooting

If you find your server running, but you cannot access it in your browser at http://0.0.0.0:3001/, try starting the server with the following command: `python run.py auto_host`. This will cause flask to define its own host and port, it may be http://127.0.0.1:5000/, but you should check the print statements in your terminal as flask will tell you at which address you can view the app now.