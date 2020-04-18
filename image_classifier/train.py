# Imports here

# General imports
import os
import argparse

# Pytorch imports
import torch
from torch import nn, optim

# training and init function imports
from functions_training import train_net
from functions_init import init_hp, init_data, init_net, init_tracker

# Classifier imports
from classifiers import SN_Classifier, DN_Classifier

#---------------------------------------------------------------------
print("-" * 100)

# Get current working directory
cwd = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument('data_directory',
                    action = 'store',
                    type = str,
                    help = 'directory of stored data'
                   )

parser.add_argument('--save_dir',
                    action = 'store',
                    default = cwd,
                    type = str,
                    help = f'directory to save checkpoints, default = {cwd}'
                   )

parser.add_argument('--fn',
                    action = 'store',
                    default = "",
                    type = str,
                    help = f'string to append to checkpoint filename, default = ""'
                   )

parser.add_argument('--arch',
                    action = 'store',
                    choices = ['squeezenet', 'densenet'],
                    default = 'squeezenet',
                    help = 'neural net architecture, choose between "squeezenet" or "densenet"'
                   )

parser.add_argument('--batch_size',
                    action = 'store',
                    default = 50,
                    type = int,
                    help = 'batch size used by the DataLoaders, default = 50'
                   )

parser.add_argument('--learning_rate',
                    action = 'store',
                    default = 0.0025,
                    type = float,
                    help = 'learning rate used by the optimizer during training, default = 0.0025'
                   )

parser.add_argument('--hidden_units_1',
                    action = 'store',
                    default = 1024,
                    type = int,
                    help = 'number of hidden units in the first layer of the classifier, default = 1024'
                   )

parser.add_argument('--hidden_units_2',
                    action = 'store',
                    default = 512,
                    type = int,
                    help = 'number of hidden units in the second layer of the classifier, default = 512'
                   )

parser.add_argument('--dropout',
                    action = 'store',
                    default = 0.3,
                    type = float,
                    help = 'Dropout probability used by the classifier during training, default = 0.3'
                   )

parser.add_argument('--epochs',
                    action = 'store',
                    default = 10,
                    type = int,
                    help = 'number of epochs to train the classifier, default = 10'
                   )

parser.add_argument('--gpu',
                    action = 'store_true',
                    help = 'use the gpu for training, default = False'
                   )

parser.add_argument('--batch_prints',
                    action = 'store_true',
                    help = 'print batch loss / accuracy in addition to validation / epoch loss / accuracy, default = False'
                   )

parser.add_argument('--plots',
                    action = 'store_true',
                    help = 'plot loss/accuracy at the end of training, default = False'
                   )

args = parser.parse_args()

# Range checks
if args.dropout and (args.dropout > 1. or args.dropout < 0.):
    parser.error("dropout must be between 0 and 1")

if args.epochs <= 0:
    parser.error("epochs must be greater than 0")

if args.batch_size <= 0:
    parser.error("batch_size must be greater than 0")

if args.learning_rate <= 0:
    parser.error("learning_rate must be greater than 0")

if args.hidden_units_1 <= 0:
    parser.error("hidden_units_1 must be greater than 0")

if args.hidden_units_2 <= 0:
    parser.error("hidden_units_2 must be greater than 0")

# Variable assignment
data_dir = args.data_directory
save_dir = args.save_dir
fn = args.fn
arch = args.arch
batch_size = args.batch_size
learning_rate = args.learning_rate
hidden_units_1 = args.hidden_units_1
hidden_units_2 = args.hidden_units_2
dropout = args.dropout
n_epochs = args.epochs
gpu = args.gpu
batch_prints = args.batch_prints
plots = args.plots

# Hidden Units Tuple
hidden_units = (hidden_units_1, hidden_units_2)

# Directory checks
if not data_dir.startswith("/"):
    data_dir = cwd + "/" + data_dir

if not save_dir.startswith("/"):
    save_dir = cwd + "/" + save_dir

# Checkpoint name and path
checkpoint_name = arch + fn + ".pth"
checkpoint_path = save_dir + "/" + checkpoint_name

# Prints
print("\n")
print(f"data_dir        = {data_dir}")
print(f"save_dir        = {save_dir}")
print(f"checkpoint_name = {checkpoint_name}")
print(f"checkpoint_path = {checkpoint_path}")
print(f"arch            = {arch}")
print(f"learning_rate   = {learning_rate}")
print(f"hidden_units_1  = {hidden_units_1}")
print(f"hidden_units_2  = {hidden_units_2}")
print(f"dropout         = {dropout}")
print(f"batch_size      = {batch_size}")
print(f"n_epochs        = {n_epochs}")
print(f"gpu             = {gpu}")
print(f"batch_prints    = {batch_prints}")
print(f"plots           = {plots}\n")
print("\n")
print("-" * 100)

# Device assignment
if gpu:
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

# Shuffle toggles for the train, valid, test datasets
shuffle = [True, True, True]

# Print Statements per validation and epoch during training
valid_prints = True
epoch_prints = True

# Classifier params for mean of weights, std of weights, and bias value initialization
init_params = [0, 0.01, 0.1]

# Initialize hyper params dict
hp = init_hp(batch_size, learning_rate, n_epochs, shuffle, hidden_units, dropout, init_params)

# Initialize data dict
data = init_data(data_dir, hp)

# Initialize tracker dict
track = init_tracker(data, hp)

# Initialize network dict
net = init_net(data, hp, arch)

# Set pretrained model's required_grad to false (freeze network params)
for param in net['model'].parameters():
    param.requires_grad = False

# After requires_grad is turned off for the pretrained model, update the model's classifier to the chosen custom classifier
if arch.lower() == "squeezenet":

    # Use Squeezenet Classifier
    net['model'].classifier = SN_Classifier(net, hp)

elif arch.lower() == "densenet":

    # Use Densenet Classifier
    net['model'].classifier = DN_Classifier(net, hp)

# After classifier is updated, create criterion and optimizer, and update net dict. Note that Optimizer is only assigned to the model.classifier
net['classifier'] = net['model'].classifier
net['criterion'] = nn.NLLLoss()
net['optimizer'] = optim.Adam(net['model'].classifier.parameters(), lr=learning_rate)

run = train_net(data, hp, track, net, device, checkpoint_path, batch_prints, valid_prints, epoch_prints, plots)
