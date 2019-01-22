# imports

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import model_utils as mu


# process an image, in order to use it in a prediction model (creates a numpy array of the image)
def process_image(image):
    '''Processes an image to be used for inference with a pytorch model.
    
    Parameters
    ----------
    image : PIL.Image
        An image opened with the PIL.Image.
    
    Returns
    -------
    ndarray
        Numpy Array describing the given image as a matrix.
    '''


    # get the width and height
    width, height = image.size
    min_side_length = 256
    crop_length = 224
    # resize the image to be 256pixels wide on the shortest side
    if width > height:
        image.thumbnail((99999, min_side_length))
    else:
        image.thumbnail((min_side_length, 99999))
    
    # crop the image
    width, height = image.size
    
    left = (width - crop_length) / 2
    top = (height - crop_length) / 2
    right = left + crop_length
    bottom = top + crop_length
    
    image = image.crop((left, top, right, bottom))
    
    # convert image to a numpy array and adjust its color scale: 0-255 (int) -> 0-1 (float)
    image = np.array(image) / 255
    
    # normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std
    
    # rearrange dimensions to have color channel as the first
    image = image.transpose(2, 0, 1)
    
    return image

# helper function to display the image in the prediction graphic
def imshow(image, ax=None, title=None):
    '''helper function to display an image in the prediction graphic
    
    Parameters
    ----------
    image : ndarray
        Image as ndarray
    ax : Axes, optional
        Pyplot axes (the default is None, which means the function will create its own subplot)
    title : str, optional
        String containing a title for the image plot (the default is None, which results in no title being added)
    
    Returns
    -------
    Axes
        Pyplot axes with the image
    '''


    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # (!) this was missing before, so I added it here to actually use the title
    ax.set_title(title)
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# visually display the prediction of a given image
def view_prediction(image, probabilities, labels, image_path=None, category_names=None):
    '''Plots image and prediction using matplotlib.
    
    Parameters
    ----------
    image : PIL.Image
        Image to be displayed.
    probabilities : Array
        Top K Probabilities to be displayed.
    labels : Array
        Top K Labels to be displayed.
    image_path : str, optional
        String describing the image location (the default is None, which results in the plot to be Untitled)
    category_names : dict, optional
        Dictionary containing the category names (the default is None, which results in the labels to be used for labeling)
    
    '''

    # Set up the plot
    figure, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    
    # assign names to labels if available
    image_title = 'Untitled'
    if category_names is not None and image_path is not None:
        labels = mu.map_label_names(labels, category_names)
        image_path_list = image_path.split('/')
        image_class_id = image_path_list[-2]
        image_title = category_names[image_class_id]
    
    # plot the image
    ax1.axis('off')
    imshow(image, ax1, image_title)
    
    # plot predictions
    y = np.arange(len(probabilities))
    ax2.barh(y, probabilities)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()