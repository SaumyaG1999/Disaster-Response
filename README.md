# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#started)
	1. [Dependencies](#dependencies)
	2. [Additional Material](#material)
3. [Authors](#authors)
4. [Acknowledgement](#acknowledgement)

## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure.
2. Machine Learning Pipeline to train a model able to classify text message in categories.
3. Web App to show model results in real time.

## Getting Started

### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Additional Material

In the **data** and **models** folder you can find two jupyter notebook that will help you understand how the model works step by step:
1. **ETL Preparation Notebook**: learn everything about the implemented ETL pipeline
2. **ML Pipeline Preparation Notebook**: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

You can use **ML Pipeline Preparation Notebook** train the model or tune it through a dedicated Grid Search .
In this case, it is warmly recommended to use a Linux machine to run Grid Search, especially if you are going to try a large combination of parameters in the model .
Use of a standard desktop/laptop may take few hours to complete. 

## Authors

* [Saumya Garg](https://github.com/SaumyaG1999)

## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a great Data Science Nanodegree Program . Grateful to the whole udacity team .
* Msut give credits to [Figure Eight](https://www.figure-eight.com/) for providing the dataset to train my model.

