# Udacity Data Science Nanodegree Capstone "Dog Breed Classifier"

## Project Overview

This is one of the capstone projects for the Udacity Data Science Nanodegree.  The goal is to train a convolutional neural network to classify dog breeds on a given picture and display the results. The data was provided by Udacity.

The projected is divided into two parts. Part 1 focused on building and training a convolutional neural network to classify dog breeds. Part 2 focused on building a simple flask app that can be used to upload an image of a dog or human and get the resembling dog breed.

### Part 1 - Building the CNN

The process is described in detail in the dog-app notebook in cnn_setup/dog_app.*, so I will not explain the entire process here in the README.

The following topics are covered in the notebook:
- The data used for the training and testing (the data was provided by Udacity)
- Building an algorithm to determine whether there is a dog in the given picture.
- Building an algorithm to determine whether there is a human face in the given picture.
- Building a CNN from scratch.
- Building a CNN based on VGG16 using transfer learning.
- Building, training and optimizing a CNN based on Resnet50 using transfer learning.
- Putting it all together in a single algorithm, that takes a picture, verifies a dog or human is in it and then returns the predicted dog breed.

The classifier currently has an accuracy of roughly 85%, which is not perfect, but regarding the difficulty of accurately determining the breed of a dog it's decent. The classifier can differ between 133 dog breeds, so the chance of guessing right is about 0.75%.

### Part 2 - Building the Flask app

As a front end for the prediction algorithm I set up a flask app. To do so I...
- set up a simple server using flask.
- set up a simple web-based frontend using bootstrap and jquery.
- transfered the prediction algorithm from the jupyter notebook to a python file
- build a flask API '/classify' which,
    - takes an image and saves it on the server (uploaded using the File API and Flask Werkzeug)
    - verifies a dog or human face is in the image
    - predicts the dog breed
    - deletes the image again (there is no point in saving the file permanently)
    - returns the prediction as JSON object

The prediction is displayed in the app - instructions on how to use it can also be found there.

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

## Further Development

The current version of the app works fine, however it is very basic - not much more than an MVP. For the scope of this project, especially regarding the limited time frame, this is enough. Nonetheless, I do have several ideas how to improve it at a later stage:
- Currently the algorithm has an accuracy of about 85%. Predicting the breed of a dog is not easy, considering that even humans have troubles differing between some breeds, however I believe that the accuracy could still be increased with more optimization or further refinement of the algorithm.
- The app currently only returns the name of the breed. The algorithm differs between 133 breeds and while I personally like dogs I have to admit I don't know many of these breeds. Therefore, I think it would be nice to also display an image of the predicted breed. Possibly using a web search and implementing it below the prediction.
- I personally find it very interesting how a neural network "sees" objects. I think it could be fun to add either functionalities like google deep dream or simply display how the neural network sees the predicted dog breed on the page as well, similar to what has been done here: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
- Currently the classifier only works for 1 human face or dog in the picture. It may be nice to expand that so it can recognize more than just one, like a dog and his/her owner. Or several dogs/human faces.
- I'm sure there is a lot more great ideas on how to improve this app, and I may get back to it after finishing this nanodegree. :)

## License

This project is distributed under the Udacity License (@see LICENSE.txt).

## Final Notes

I'd like to thank Udacity for this course. The structure of the lectures and the projects are great. When I started this nanodegree the topic seemed far away and daunting to me, but due to this program I've gotten an easy way into the field of machine learning.

What I found most challenging about this project was actually handling the File API and making sure the file is saved properly on the server. :D I guess it's been a while since I did web development, haha. It was a cool project though. I also liked some of the other potential capstone projects - especially the spark project. However, neural networks are a really interesting field and there is so much you can do with them. The best part of the project for me was learning how a convolutional neural network "sees the world". This  is definitely a topic I'll look into in more detail after the Nanodegree program.
