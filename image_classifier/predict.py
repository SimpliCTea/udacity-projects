# imports
import argparse
import json

import torch

from PIL import Image

import model_utils as mu
import image_utils as iu

parser = argparse.ArgumentParser()

parser.add_argument('image_path',   
    help='Path to the image you want to classify.')
parser.add_argument('checkpoint_path', 
    help='Path to the checkpoint for the classification model.')
parser.add_argument('--top_k', 
    type=int,
    default=1, 
    help='Number of top most likely classes to return. Default: 1')
parser.add_argument('--category_names',
    help='Path to JSON dictionary to map the classes to real names.')
parser.add_argument('--verbosity', 
    type=int, 
    default=1, 
    help='Set the verbosity for the training. 0: print nothing; 1: '
    'print basic summaries; 2: print regular information. Default: 1')
parser.add_argument('--gpu', 
    default=False, 
    action='store_true', 
    help='Use GPU for inference. Default: False')
#parser.add_argument('--show_prediction', 
#    default=False, 
#    action='store_true', 
#    help='Show a graphical representation of the prediction. Default: False')


args = parser.parse_args()

def print_architecture():
    print('Argument Setup:')
    print('-' * 30)
    print('Image Path: {}'.format(args.image_path))
    print('Checkpoint Path: {}'.format(args.checkpoint_path))
    print('Top K: {}'.format(args.top_k))
    print('Category Name File: {}'.format(args.category_names))
    print('Verbosity: {}'.format(args.verbosity))
    print('Use GPU: {}'.format(args.gpu))
#    print('Show Prediction: {}'.format(args.gpu))
    print('-' * 30)
    print('')

def print_prediction(top_probs, top_labs, top_cats = None):
    for i in range(len(top_probs)):
        result = '{}. label: {} - with a probability of {:.2f}%'.format(i+1, top_labs[i], top_probs[i]*100)
        if top_cats is not None:
            result += ' - Predicted flower name: {}.'.format(top_cats[i])
        print(result)
        if i == 0:
            print('-' * 30)

# main method
def main():
    '''Classifies a given image based on a given model.
    
    '''

    if args.verbosity >= 2:
        print_architecture()
    if args.verbosity >= 1:
        print('-' * 30)
        print('Starting inference now...')
        print('')

        
    if args.gpu and torch.cuda.is_available:
        if args.verbosity >= 1:
            print('Using GPU for training.')
        device = torch.device('cuda:0')
    elif args.gpu:
        print('Warning: GPU use requested but not available. Using CPU for training.')
        device = torch.device('cpu')
    else:
        if args.verbosity >= 1:
            print('Using CPU for training.')
        device = torch.device('cpu')
    
    # load the category names if there is a dictionary provided
    category_names = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            category_names = json.load(f)

    # load and process the image for inference
    image = Image.open(args.image_path)
    image = iu.process_image(image)

    # create a model from the given checkpoint
    model = mu.load_checkpoint(args.checkpoint_path)

    # do the prediction
    top_probabilities, top_labels = mu.predict(image, model, topk=args.top_k, device = device)

    if category_names is not None:
        top_flowers = mu.map_label_names(top_labels, category_names)
        print_prediction(top_probabilities, top_labels, top_flowers)
#        if args.show_prediction:
#            iu.view_prediction(image, top_probabilities, top_labels, image_path=args.image_path, category_names=category_names)
    else:
        print_prediction(top_probabilities, top_labels)
#        if args.show_prediction:
#            iu.view_prediction(image, top_probabilities, top_labels)

if __name__ == "__main__":
    main()