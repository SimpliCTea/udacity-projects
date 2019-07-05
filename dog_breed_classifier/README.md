# Udacity Data Science Nanodegree Capstone "Dog Breed Classifier"

## Project Overview

The projected is divided into two parts. Part 1 focused on building and training a convolutional neural network to classify dog breeds. Part 2 focused on building a simple flask app that can be used to upload an image of a dog or human and get the resembling dog breed.

### Part 1 - Building the CNN

The process is described in detail in the dog-app notebook in cnn_setup/dog_app.*, so I will not explain the entire process here in the README.

The following topics are covered in the notebook:
- The data used for the training and testing
- Building an algorithm to determine whether there is a dog in the given picture.
- Building an algorithm to determine whether there is a human face in the given picture.
- Building a CNN from scratch.
- Building a CNN based on VGG16 using transfer learning.
- Building, training and optimizing a CNN based on Resnet50 using transfer learning.
- Putting it all together in a single algorithm, that takes a picture, verifies a dog or human is in it and then returns the predicted dog breed.

### Part 2 - Building the Flask app

As a front end for the prediction algorithm I set up a flask app. To do so I...
- set up a simple server using flask.
- set up a simple web-based frontend using bootstrap and jquery.
- transfered the prediction algorithm from the jupyter notebook to a python file
- build a flask API '/classify' which,
    - takes an image and saves it on the server
    - verifies a dog or human face is in the image
    - predicts the dog breed
    - deletes the image again (there is no point in saving the file permanently)
    - returns the prediction as JSON object
- the prediction is displayed in the app; instructions on how to use it can also be found there

## Requirements

This project uses python 3.5, keras, tensorflow, numpy and flask. Regarding the specific requirements and instructions on how to use them I'll kindly refer you to the README in ./cnn_setup and the folder ./cnn_setup/requirements. There you'll find a requirements file and several environment.yml that can be used to set up a virtual environment with conda.

The README and the requirement/environment files were provided by Udacity. The only change on my side was adding Flask, which I used to build a server and front end for the classifier. I've already added the flask requirement to all the files.

I developed and tested on Mac and it should be running fine on MacOS. However, when testing on Windows I've run into a numpy import error. If you encounter the same problem, you may want to update numpy. That fixed the problem for me. You can do so by using `pip install -U numpy`. Be sure to have the new environment active when doing so.

## Instructions

Once all requirements are met and you activated the virtual environment, you can run the app following these instructions:

- move to the folder "flask_app": `cd path/to/flask_app`
- set the environment variable FLASK_APP to run.py
    - Mac / Linux: `export FLASK_APP=run.py`
    - Windows: `set FLASK_APP=run.py`
- start the server with: `flask run --without-threads`- the --without-threads ensures that the server is not doing stuff on parallel threads on your cpu when it shouldn't
- open a web browser and enter the address given in the shell (it was 127.0.0.1:5000 for me, but may be different for you so better check the shell!)
- further instructions on how to use the app are described on the starting page

If you are interested in how the CNN was trained, checkout the jupyter notebook at cnn_setup/dog_app.ipynb or the html version of it at cnn_setup/dog_app.html. There you'll find the entire process explained with coding examples.

## License

This project is distributed under the Udacity License (@see LICENSE.txt).