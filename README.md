# Disaster Response Pipeline 

## Table of contents
1. Introduction
2. Installation
3. Instructions

# 1. Introduction
This project demonstrates how to make an ETL pipeline, Machine Learning pipeline and finaly a web app. It uses (social) media data scaped from the web to predict if there is a help post between them.

# 2. Installation
This project is written in Python 3.6. Make sure that the folowing libraries are installed:
- pandas
- sqlalchemy 
- sklearn
- nltk
- flask

# 3. Instructions:
1. First run the ETL pipeline to make a SQLite .db file from the raw events\
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Run the ML pipeline to make a pkl file with the trained model\
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Then run the webserver in the app directory  with the folowing command\
`python app/run.py`

3. Go to http://0.0.0.0:3001/ on Linux or http://127.0.0.1:3001 on Windows for the webpage

