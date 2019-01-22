# imports
import argparse
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import model_utils as mu

parser = argparse.ArgumentParser()
# add all the necessary arguments
parser.add_argument('data_dir',
    help='Directory of the images to train, validate and test the model with. '
    'Should be structered into the subfolders: train, valid and test.')
parser.add_argument('--save_dir',
    default='./', 
    help='Directory for the checkpoints to be saved into. Default: same directory as the train.py file')
parser.add_argument('--arch',
    default='vgg19', 
    help='Architecture of the model. Supported architectures are: vgg16, vgg19, '
    'densenet-161, densenet-201, resnet-101, resnet-152. Default: vgg19')
parser.add_argument('--learning_rate',
    type=float, 
    default=0.001, 
    help='Set learning rate for training the model. Default: 0.001')
parser.add_argument('--hidden_units',
    type=int, 
    default=512, 
    help='Set the number of units for the hidden layer. Default: 512.')
parser.add_argument('--epochs', 
    type=int, 
    default=10, 
    help='Set the number of epochs used for training the model. Default: 10')
parser.add_argument('--verbosity', 
    type=int, 
    default=1, 
    help='Set the verbosity for the training. 0: print nothing; 1: print basic information; '
    '2: print regular and more detailed information. Default: 1')
parser.add_argument('--gpu', 
    default=False, 
    action='store_true', 
    help='Use GPU for inference. Default: False')
parser.add_argument('--check_accuracy', 
    default=False, 
    action='store_true', 
    help='Check and print accuracy on testing set after training the model. Default: False')

args = parser.parse_args()

# print architecture if verbosity is high
def print_architecture():
    print('Argument Setup:')
    print('-' * 30)
    print('Data Directory: {}'.format(args.data_dir))
    print('Save Directory: {}'.format(args.save_dir))
    print('Architecture: {}'.format(args.arch))
    print('Learning Rate: {}'.format(args.learning_rate))
    print('Hidden Units: {}'.format(args.hidden_units))
    print('Epochs: {}'.format(args.epochs))
    print('Verbosity: {}'.format(args.verbosity))
    print('Use GPU: {}'.format(args.gpu))
    print('Check accuracy after training: {}'.format(args.check_accuracy))
    print('-' * 30)
    print('')

# main method, which does the training
def main():
    '''Trains a model and saves it to a checkpoint. Uses Pytorch.
    
    '''

    if args.verbosity >= 2:
        print_architecture()

    batch_size = 32

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

    # prepare dataset
    image_datasets, dataloaders, dataset_sizes = mu.prepare_datasets(args.data_dir, batch_size=batch_size)

    # load the correct model and set it up
    model, input_size = mu.load_model(args.arch)

    for param in model.parameters():
        param.requires_grad = False

    classifier_structure = OrderedDict([
        ('fc1', nn.Linear(input_size, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ])

    model.classifier = nn.Sequential(classifier_structure)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # train the model and save the checkpoint
    model = mu.train_model(model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epochs=args.epochs, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device, verbosity = args.verbosity)

    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'architecture': args.arch,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    checkpoint_path = args.save_dir + 'checkpoint_cmd_' + time.strftime('%Y%m%d_%H%M%S') + '.pth'
    torch.save(checkpoint, checkpoint_path)

    if args.check_accuracy:
        mu.check_accuracy(model, dataloaders['test'], device=device, verbosity=args.verbosity)

if __name__ == "__main__":
    main()