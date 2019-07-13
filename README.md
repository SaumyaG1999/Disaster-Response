# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Guide](#Guide)
	1. [Dependencies](#dependencies)
	2. [Executing Program](#Executing)
	3. [Additional Material](#material)
3. [Authors](#Authors)
4. [Acknowledgements](#Acknowledgements)

## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in 3 Sections:

1. Data Pre-Processing which includ an ETL Pipeline to extract data from source, clean data and save them in a proper databse structure.
2. Machine Learning Pipeline to train a model, able to classify text message in categories.
3. Web App to show model results in real time.

## Guide:

### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Executing Program

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Go to http://0.0.0.0:3001/

### Additional Notebook's:

In the **data** and **models** folder, you can find one jupyter notebook each that will help you understand how the model works step by step:
1. **ETL Pipeline Preparation Notebook**: learn everything about the implemented ETL pipeline
2. **ML Pipeline Preparation Notebook**: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn libraries

You can use **ML Pipeline Preparation Notebook** to train the model or tune it with the help of a Grid Search .
In this case, it is recommended to use a machine with GPU to run Grid Search, especially if you are going to try a large combination of parameters in the model .
Use of a standard desktop/laptop may take few hours to complete. 

## Authors

* [Saumya Garg](https://github.com/SaumyaG1999)

## Acknowledgements

* Must give credits to [Udacity](https://www.udacity.com/) for providing such a great Data Science Nanodegree Program . Grateful to the   whole udacity team .
* [Figure Eight](https://www.figure-eight.com/) for providing the dataset to train my model.

