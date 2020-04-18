# Imports here

# General imports
import matplotlib.pyplot as plt
import numpy as np

# Pytorch imports
from torch.utils.data import DataLoader

# Torchvision imports
from torchvision.datasets import ImageFolder
from torchvision import models, transforms


# Init Hyper Parameters dict
def init_hp(batch_size, learn_rate, n_epochs, shuffle, hidden_units, p_dropout, init_params):
    ''' Function to initialize the Hyper Parameters Dict

        Args:
            batch_size (int): Number of data samples per batch
            learn_rate (float): Scalar to adjust the magnitude of the changes when updating weights in a neural net
            n_epochs (int): Number of times to iterate over the entire training dataset
            shuffle (list of booleans): Toggle shuffle for the data batches
            hidden_units (tuple): Tuple of Ints containing number of hidden units in first and second hidden layers
            p_dropout (float): Between 0 and 1. Probability of "dropping out" any given unit within a layer
            init_params (list of floats): Params passed to initialize the Classifier Weights and Bias
                Float 1: mean of the normal distribution of the Classifier Weights
                Float 2: std of the normal distribution of the Classifier Weights
                Float 3: Constant of the bias value of the Classifier Weights

        Returns:
            hp (dict): Dict containing all of the hyper parameters defined above
    '''
    hp = {}

    hp = {'batch_size': batch_size, 'learn_rate': learn_rate, 'n_epochs': n_epochs, 'shuffle': shuffle, 'hidden_units': hidden_units, 'p_dropout': p_dropout, 'init_params': init_params}

    return hp


# Init Data dict
def init_data(data_dir, hp):
    ''' Function to initialize the Data Dict using preset transforms. The Data is expected to be pre-split into training, validation, and test sets named 'train', 'valid', and 'test' respectively. Within these, the data is expected to be pre-split into separate folders, each representing a class.

        Args:
            data_dir (string): directory containing the data
            hp (dict): the initialized hyper params dict. This is used to get the desired batch_size and shuffle status

        Returns:
            data (dict): ['train', 'valid', 'test']
                dirs (string): data directories
                transforms: Transforms objects
                dataset: DataSet objects
                dataloader: DataLoader objects
                n_images (int): number of images
                n_batches (int): number of batches
                n_classes (int): number of classes
    '''

    data = {}
    data = {'train': {}, 'valid': {}, 'test': {}}
    
    # Batch size and Shuffle booleans to use with DataLoader
    batch_size = hp['batch_size']
    shuffle = hp['shuffle']

    # Create data directories per data set
    for d in data:

        data[d] = {'dirs': data_dir + "/" + d, 'transforms': None, 'dataset': None, 'dataloader': None, 'n_images': None,
                   'n_batches': None, 'n_classes': None}

    # Define data transforms per data set. All data sets are transformed to tensors and normalized by the mean and std values given in the Pytorch Pretrained Nets documentation

    # Training Data is randomly rotated, resized/cropped, and flipped.
    data['train']['transforms'] = transforms.Compose([transforms.Resize(255),
                                  transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])

    # Validation and testing datasets are center cropped, no random transformations are applied
    data['valid']['transforms'] = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])

    data['test']['transforms'] = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])

    # Create DataSets via ImageFolder, and DataLoaders
    for i, d in enumerate(data):

        # DataSets, number of images, and number of classes
        data[d]['dataset'] = ImageFolder(data[d]['dirs'], data[d]['transforms'])
        data[d]['n_images'] = int(len(data[d]['dataset']))
        data[d]['n_classes'] = int(len(data[d]['dataset'].classes))

        # DataLoaders, number of batches
        data[d]['dataloader'] = DataLoader(data[d]['dataset'], batch_size = batch_size, shuffle = shuffle[i])
        data[d]['n_batches'] = int(len(data[d]['dataloader']))

    return data


# Init Net dict
def init_net(data, hp, arch):
    ''' Function to initialize the Net Dict. This is used to store Model information which is saved within checkpoints.

        Args:
            data (dict): Data Dict. This is used to get the number of classes within the dataset, which is stored within the Net Dict
            hp (dict): Hyper Params Dict. This is used to get the tuple of desired hidden units in the first two layers
            arch (string): the name of the desired pretrained net architecture to load

        Returns:
            net (dict):
                classifier: None (not defined in this function)
                in_features (int): 512 for squeezenet, 1024 for densenet
                hidden_units (tuple): number of desired hidden units in the first 2 layers of the classifier
                out_features (int): out_features of last layer of classifier
                criterion: Loss function (not defined in this function)
                optimizer: Optimizer to update weights of classifier (not defined in this function)
                model: pretrained model (squeezenet or densenet)
                model_name (string): "squeezenet" or "densenet"
    '''
    
    # Get the hidden_units tuple from the Hyper Params dict
    hidden_units = hp['hidden_units']

    net = {}
    
    # Initialize Net Dict
    net = {'classifier': None, 'in_features': None, 'hidden_units': hidden_units, 'out_features': None, 'criterion': None, 'optimizer': None, 'model': None, 'model_name': None}
    
    # out_features of last layer = number of classes in dataset
    net['out_features'] = data['train']['n_classes']

    if arch.lower() == "squeezenet":
        
        # Download pretrained squeezenet model
        net['model'] = models.squeezenet1_0(pretrained=True)
        
        # Squeezenet in_features = in_channels = 512
        net['in_features'] = 512

        # Number of classes needs to be explicitly changed for squeezenet model from 1000 to out_features
        net['model'].num_classes = net['out_features']

    elif arch.lower() == "densenet":
        
        # Download pretrained densenet model
        net['model'] = models.densenet121(pretrained = True)
        
        # Densenet in_features = 1024
        net['in_features'] = 1024

    # Get model name
    net['model_name'] = net['model'].__class__.__name__.lower()

    return net

# Init tracker dict
def init_tracker(data, hp):
    ''' Function to initialize the tracker dict. This is used to track the stats of a network during training, validation, and testing

        Args:
            data (dict): Data dict. This is used to get the number of training, validation, and testing batches
            hp (dict): Hyper Params dict. This is used to get the number of training epochs

        Returns:
            tracker (dict):
                train:
                    batch_loss / batch_acc (numpy arrays): Arrays to store all batch loss/acc values per training batch per epoch
                    epoch_loss / epoch_acc (numpy arrays): Arrays to store epoch loss/acc values per epoch
                    av_loss / av_acc (numpy arrays): Arrays to store average training loss/acc values per validation

                valid:
                    batch_loss / batch_acc (numpy arrays): Arrays to store all batch loss/acc values per validation batch per validation
                    av_loss / av_acc (numpy arrays): Arrays to store avergae validation loss/acc values per validation

                test:
                    batch_loss / batch_acc (numpy arrays): Arrays to store all batch loss/acc values per testing batch
                    av_loss / av_acc (floats): Floats to store testing loss/acc values per test (once)

    '''

    # Init track dict
    track = {'train': {}, 'valid': {}, 'test': {}}

    # Number of Epochs
    n_epochs = hp['n_epochs']

    # Number of training and total training batches
    n_train_batches = data['train']['n_batches']
    n_total_train_batches = n_epochs * n_train_batches

    # Number of validation and total validation batches, and number of validations across all epochs
    n_valid_batches = data['valid']['n_batches']
    n_valids = int(np.floor(n_total_train_batches / n_valid_batches))
    n_total_valid_batches = n_valids * n_valid_batches

    # Number of test batches = Number of total test batches since the testing dataset is only evaluated once
    n_test_batches = data['test']['n_batches']

    for d in track:

        if d == 'train':

            track[d]['batch_loss'] = np.ones((n_total_train_batches)) - 100
            track[d]['batch_acc'] = np.ones((n_total_train_batches)) - 100
            track[d]['epoch_loss'] = np.ones((n_epochs)) - 100
            track[d]['epoch_acc'] = np.ones((n_epochs)) - 100
            track[d]['av_loss'] = np.ones((n_valids)) - 100
            track[d]['av_acc'] = np.ones((n_valids)) - 100

        if d == 'valid':

            track[d]['batch_loss'] = np.ones((n_total_valid_batches)) - 100
            track[d]['batch_acc'] = np.ones((n_total_valid_batches)) - 100
            track[d]['av_loss'] = np.ones((n_valids)) - 100
            track[d]['av_acc'] = np.ones((n_valids)) - 100

        if d == 'test':

            track[d]['batch_loss'] = np.ones((n_test_batches)) - 100
            track[d]['batch_acc'] = np.ones((n_test_batches)) - 100
            track[d]['av_loss'] = -99
            track[d]['av_acc'] = -99

    return track
