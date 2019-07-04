# imports
import os
import pickle
import cv2
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import model_from_json

from PIL import Image, ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True

from app.extract_bottleneck_features import extract_Resnet50

def testf():
    print('Test successful!')

def get_path(filename):
    """Grabs the absolute path to the file in the models directory.
    
    Returns:
        string -- path to file in models directory
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'static/models/')
    path = os.path.join(path, filename)
    return path

def load_breeds():
    """Loads a list of breed names for the classifier.
    
    Returns:
        list -- List of breed names in same order as the classifier labels.
    """
    pkl_path = get_path('breednames.pkl')
    with open(pkl_path, 'rb') as pkl:
        breed_names = pickle.load(pkl)
    return breed_names

def detect_face(img_path):
    """Loads given image and attempts to detect human faces.
    
    Arguments:
        img_path {string} -- Path to the image that should be analysed.
    
    Returns:
        boolean -- True if a human face was detected.
    """
    clf_path = get_path('haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier(clf_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def img_to_tensor(img_path):
    """Loads given image and transforms it into 4D tensor.
    
    Arguments:
        img_path {string} -- Path to the image to be transformed
    
    Returns:
        numpy array -- 4D tensor of the image with shape (1, 224, 224, 3)
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def detect_dog(img_path):
    """Loads  given image and attempts to detect a dog.
    
    Arguments:
        img_path {string} -- Path to image to be analysed.
    
    Returns:
        boolean -- True if dog was detected.
    """
    model = ResNet50(weights='imagenet')
    img = preprocess_input(img_to_tensor(img_path))
    prediction = np.argmax(model.predict(img))
    # dogs are between 151th and 268th label of the model
    return ((prediction <= 268) & (prediction >= 151))

def load_breed_classifier():
    """Loads the trained breed classifier.
    
    Returns:
        keras model -- Breed classifier based on Resnet50
    """
    # load json and create model
    clf_arch_path = get_path('breed_clf_architecture.json')
    with open(clf_arch_path, 'r') as json_file:
        model_json = json_file.read()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    clf_weight_path = get_path('breed_clf_weights.hdf5')
    loaded_model.load_weights(clf_weight_path)
    return loaded_model

def predict_breed(img_path):
    """Loads classifier and does prediction on given image.
    
    Arguments:
        img_path {string} -- Path to image for the prediction
    
    Returns:
        string -- Name of the predicted breed.
    """
    # load model
    model = load_breed_classifier()
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(img_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    breed_names = load_breeds()
    return breed_names[np.argmax(predicted_vector)]

def predict(img_path):
    """Takes image path and tries to predict whether it's a dog or human and the resembling dog breed.
    
    Arguments:
        img_path {string} -- Path to image for the prediction
    
    Returns:
        dict -- A dictionary containing two keys: race (dog/human) and breed (resembling breed)
    """
    is_dog = detect_dog(img_path)
    is_human = detect_face(img_path)
    # make sure that the image is of a dog or a human face
    assert(is_dog | is_human), 'The algorithm only accepts images of dogs or human faces.'
    # return both the type (dog/human) and the predicted breed
    prediction = {
        'race': 'dog' if is_dog else 'human',
        'breed': ''
    }
    prediction['breed'] = predict_breed(img_path)
    return prediction