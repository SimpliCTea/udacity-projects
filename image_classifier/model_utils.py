#imports
import time, copy

from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms

from PIL import Image

from workspace_utils import active_session

# Get a basic setup for the neural network
def prepare_datasets(data_dir, batch_size = 32):
    """Creates and returns a basic setup for a neural network in pytorch
    
    Parameters
    ----------
    data_dir : String
        Path of the image root folder. The folder should contain the 3 subfolders for train, validation (= valid) and test
    batch_size : Integer    
        Number of images per batch.
    
    Returns
    -------
    Tuple
        Contains 3 dictionaries with (in order) image datasets, dataloaders and dataset sizes; each dictionary contains the keys for the three datasets: 'train', 'valid' and test'.
    """

    set_dir = {
        'train': data_dir + '/train',
        'valid': data_dir + '/valid',
        'test': data_dir + '/test'
    }

    data_transforms = {}
    data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30), 
                                        transforms.RandomResizedCrop(224), 
                                        transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    data_transforms['valid'] = transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    data_transforms['test'] = transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {set_name: datasets.ImageFolder(set_dir[set_name], data_transforms[set_name]) for set_name in [*set_dir]}


    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {set_name: torch.utils.data.DataLoader(image_datasets[set_name], batch_size=batch_size, shuffle=True) for set_name in [*set_dir]}

    # save the sizes
    dataset_sizes = {set_name: len(image_datasets[set_name]) for set_name in [*set_dir]}

    return image_datasets, dataloaders, dataset_sizes

def map_label_names(labels, category_names):
    '''Maps category names to the labels.
    
    Parameters
    ----------
    labels : Array
        List of labels.
    category_names : Array
        List of names. Should have the same number of names as labels in the list of labels.
    
    Returns
    -------
    Array
        List of category names in appropriate order.
    '''

    return [category_names[label] for label in labels]

# TODO: Either add class_map as input or make a new function for that as you need it in image_utils as well
def predict(image, model, topk=5, device=None):

    '''Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters
    ----------
    image : ndarray
        A numpy array describing the image you want to predict.
    model : Pytorch Model
        [description]
    topk : int, optional
        Number of top classes the model predicted for the image. (the default is 5, which means the top 5 classes will be returned)
    device : torch.device
        A torch device to use for the prediction: cpu or cuda. Default: None, which results in using cpu.
    
    Returns
    -------
    Tuple
        Tuple with 2 arrays: 1. probabilities for the top k classes; 2. top k classes
    '''

    
    if device == None:
        device = torch.device('cpu')
    
    model.to(device)
    # Predict the class (or classes) of an image using a trained deep learning model.
    
    # image processing method returns a numpy array, but for torch we need a tensor
    if device == torch.device('cpu'):
        image = torch.from_numpy(image).type(torch.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    
    # torch was complaining about wrong batch size; solution taken from here: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    image = image.unsqueeze(0)
    
    # send image to the correct device as well
    image.to(device)
    # get the predictions
    prediction = model(image)
    # this will return the logarithmic predictions as the last output layer is with LogSoftmax
    prediction = torch.exp(prediction)
    
    # get the topk predictions
    top_probabilities, top_labels_idx = prediction.topk(topk)
    
    # got an error with moving over the tensors using the dictionary methods below; so I'll turn them back into numpy arrays
    # which caused another error and suggested to use 'var.detach().numpy()'
    # which caused another error as they were not 'hashable'; rather than using numpy arrays I'll convert them into lists
    # trying the prediction with gpu caused another few errors; in case of gpu usage the FloatTensors are now 
    # cuda.FloatTensors and have to be brought back to cpu before turning them into numpy arrays 
    
    top_probabilities = top_probabilities.detach().cpu().numpy()[0].tolist()
    top_labels_idx = top_labels_idx.detach().cpu().numpy()[0].tolist()
    
    # swap the class_to_idx, cause we need the class now with the cat_to_name
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    # get the correct labels
    top_labels = [idx_to_class[label] for label in top_labels_idx]
    
    return top_probabilities, top_labels

# loads the model from a checkpoint
def load_checkpoint(checkpoint_path):
    '''Loads a torch checkpoint and rebuilds the model.
    
    Parameters
    ----------
    checkpoint_path : str
        String describing the path to the checkpoint file
    
    Returns
    -------
    torch model
        Torch model to use in further training or for inference.
    '''

    cp = torch.load(checkpoint_path)
    #cp = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) from https://discuss.pytorch.org/t/problem-loading-model-trained-on-gpu/17745/2
    
    # load the architecture
    model, input_sizes = load_model(cp['architecture'])
    for param in model.parameters():
        param.requires_grad = False
        
    # load up class to idx
    model.class_to_idx = cp['class_to_idx']
    
    # rebuild classifier
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, cp['hidden_units'])),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(cp['hidden_units'], 102)),
                                ('output', nn.LogSoftmax(dim=1))
    ]))
    # attach the classifier to the model
    model.classifier = classifier
    
    model.load_state_dict(cp['state_dict'])
    
    return model

# Add the number of input features for all the models
def load_vgg16(pretrained=True):
    input_size = 25088
    return models.vgg16(pretrained=pretrained), input_size
def load_vgg19(pretrained=True):
    input_size = 25088
    return models.vgg19(pretrained=pretrained), input_size
def load_densenet161(pretrained=True):
    input_size = 2208
    return models.densenet161(pretrained=pretrained), input_size
def load_densenet201(pretrained=True):
    input_size = 1920
    return models.densenet201(pretrained=pretrained), input_size
def load_resnet101(pretrained=True):
    input_size = 2048
    return models.resnet101(pretrained=pretrained), input_size
def load_resnet152(pretrained=True):
    input_size = 2048
    return models.resnet152(pretrained=pretrained), input_size

# method to load the required model (works similar to switch/case statements in other languages)
def load_model(architecture, pretrained = True):
    '''Loads a torchvision model.
    
    Parameters
    ----------
    architecture : str
        String describing the model to be loaded. Valid strings are: vgg16, vgg19, densenet-161, densenet-201, resnet-101, resnet-152.
    pretrained : bool, optional
        Whether the model should be loaded in a pretrained state. (the default is True)
    
    Returns
    -------
    torch model, int
        Returns the fully loaded model and the input size for the associated classifier.
    '''

    switcher = {
        'vgg16': load_vgg16,
        'vgg19': load_vgg19,
        'densenet-161': load_densenet161,
        'densenet-201': load_densenet201,
        'resnet-101': load_resnet101,
        'resnet-152': load_resnet152
    }

    loader = switcher.get(architecture, None)

    if loader == None:
        print('Given architecture: "{}" not supported. Now defaulting to: vgg19. '
        'Valid architectures are: vgg16, vgg19, densenet-161, densenet-201, resnet-101, resnet-152.'.format(architecture))
        loader = switcher.get('vgg19', None)

    return loader(pretrained)

# method built following this tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#training-the-model
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler = None, epochs = 10, device=None, verbosity = 1):
    if verbosity >= 1:
        print('-' * 30)
        print('Starting training now...')
        print('')

    if device == None:
        if args.verbosity >= 1:
            print('Using CPU for training.')
        device = torch.device('cpu')

    # send model to the right device
    model.to(device)

    with active_session():
        start_time = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        print_every = 50

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-' * 30)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    if scheduler is not None:
                        scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                step = 0

                for images, labels in dataloaders[phase]:
                    images, labels = images.to(device), labels.to(device)
                    
                    step += 1

                    # never forget to zero gradients
                    optimizer.zero_grad()

                    # move forward through the network
                    # only track history when training the model
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        pred_probabilities, pred_classes = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # propagate the loss back through the network in training phase & update weights accordingly
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(pred_classes == labels.data)
                        
                    if verbosity >= 2 and (step % print_every) == 0: 
                        processed_images = step * dataloaders[phase].batch_size
                        print('Batch: {}, Running Loss: {:.3f}, Running Corrects: {}, Processed Images: {}/{}, Current Accuracy: {:.3f}'.format(
                            step, running_loss / processed_images, running_corrects, processed_images, dataset_sizes[phase], running_corrects.double() / processed_images))
                        
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                if verbosity >= 1:
                    print('-' * 50)
                    print('{} Loss: {:.3f} Acc: {:.3f}'.format(phase.title(), epoch_loss, epoch_acc))
                    print('-' * 50)
                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            if verbosity >= 1:
                print('')

        time_elapsed = time.time() - start_time
        if verbosity >= 1:
            print('-' * 30)
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:.3f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)

    return model

def check_accuracy(model, dataloader, device=None, verbosity = 1):
    '''Checks the accuracy of a given model based on the given dataset. Don't use this on the training set.
    
    Parameters
    ----------
    model : torch model
        Torch model to be used.
    dataloader : torch dataloader
        Dataloader containing the image to check the model with.
    device : torch device, optional
        Specifies the device to use for the inference (the default is None, which makes the method use cpu)
    verbosity : int, optional
        Set the verbosity. A high verbosity will return the accuracy for every single batch. (the default is 1, which will print the basic summary of the prediction)
    
    '''


    if verbosity >= 1:
        print()
        print('-' * 30)
        print('Testing accuracy now...')
    # set model into evaluation mode
    model.eval()

    if device == None:
        device = torch.device('cpu')
    # make sure the model is on the right device
    model.to(device)

    # do the tests without autograd
    total_test_acc = 0
    step = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            step += 1
            # get outputs of the model
            outputs = model(images)
            # get the maximum probabilities and values
            pred_probabilities, pred_classes = torch.max(outputs, dim = 1)

            accuracy = (pred_classes == labels.data).float().mean()

            if verbosity >= 2:
                print('Test accuracy for batch {} is: {:.3f}%'.format(step, accuracy.item() * 100))

            total_test_acc += accuracy.item()
    
    print()
    print('Test accuracy for the entire testing set was: {:.3f}%'.format((total_test_acc / step) * 100))


