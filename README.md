# udacity-projects
This repository holds a collection of project files for my Udacity courses.

# Requirements

All of the projects currently in this repository are part of the Become a Data Science nanodegree.

Most of these projects are completed in a Jupyter Notebook using python and several python modules such as pandas, numpy, matplotlib, seaborn, pytorch and many more. For detailed information on which modules are used check the the import statements at the top of the notebooks.

# Project Outline

I'll give a brief description of all the projects in this repository. For more details feel free to check the code or notebooks, I usually try to explain what I'm doing as comments in the code and notebook.

## Project 1: Finding Donors

Using supervised learning algorithms, I had to build a model that would select potential donors for a fictional charity organisation based on census data. Three particular algorithms were tried out: k-nearest neighbours, random forests and gradient boosting.

## Project 2: Image Classifier

This project focused on deep learning. It is structured into two parts.

Part 1 is a jupyter notebook in which a neural network is built up, which can distinguish between 102 different kinds of flowers.

Part 2 takes the code from the notebook and creates a command line program that let's you train the model and also provide an image of a flower to the model and get the type of flower back.

## Project 3: CRISP-DM & BLOGPOST

Based on the yearly stackoverflow survey I analyze various regional differences and how they develop over time. The datasets can be found here: https://insights.stackoverflow.com/survey. They are also included in the directory of this project.

For the analysis I follow the CRISP-DM process. Starting by developing a business and data understanding about the surveys, which I was previously unfmailiar with. Then, preparing the data for the analysis. And finally the analysis itself. For the analysis I provided various methods and procedures that I could use repeatedly to analyse and visualize the data. During this process I often had to iterate and go back to the data preparation tasks and even to the data understanding. Thus, this last part encompasses the Modelling, Evaluation and Deployment of the CRISP-DM.

The results were published in form of a blogpost on Medium, which can be found here: https://medium.com/@marcleonroemer/a-story-into-the-stackoverflow-yearly-survey-data-jungle-9f61d31f0f98.

## Project4: Disaster Response Pipeline

This project focuses on building Extract-Transform-Load (ETL) pipelines to automate data processing and Machine Learning (ML) pipelines to automate model building and optimization. In addition, it touches building data dashboards and web-apps using Flask and Plotly.js. 

As part of this project a dataset provided by Figure Eight (https://www.figure-eight.com/) with short messages sent in areas of crisis is analysed. With the ETL-pipeline the datasets are extracted from the given CSV files, cleaned and then loaded into a SQLite database. The ML-pipeline takes the data from the database, runs it through Natural Language Processing (NLP) and then feeds it to a machine learning model. To optimize the parameters for this pipeline a grid search is be performed. The resulting model is used in a slim web-app that gives an overview of the dataset and allows a user to enter a message and have it classified to one of several categories. This would allow helpers to immediately understand the topics of a message, such as water or medical help, to realize quickly what kind of help is required.

For more details check the Readme in the project's directory.

# License

Well, this repository is mostly used for the submission of these projects. The general outline of the projects (often icl. the layout of the jupyter notebook) usually comes from udacity. Much of the code is written by myself and you can use it in whatever way pleases you. If you are doing the same or similar course and use these project files for help, please make sure you follow the Udacity guidelines on using someone else's code.


