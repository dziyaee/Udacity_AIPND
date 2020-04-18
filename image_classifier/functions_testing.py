# Imports here

# General imports
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from PIL import Image

# Pytorch imports
import torch

from functions_training import fpass

# Load checkpoint function
def load_checkpoint(checkpoint_path):
    ''' Function to load a checkpoint dict, and load the saved state dict into the model stored within the checkpoint dict

        Args:
            checkpoint_path (string): the saved checkpoint path

        Returns:
            checkpoint (dict) with model after loading saved state dict in
    '''
    # Use current available device for loading checkpoint
    if torch.cuda.is_available():

        map_location=lambda storage, loc: storage.cuda()
    else:

        map_location='cpu'

    print(f"map_location = {map_location}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load Saved State dict from checkpoint into saved model from checkpoint
    checkpoint['net']['model'].load_state_dict(checkpoint['state_dict'])

    return checkpoint


def json_maps(json_path, class_to_idx):
    ''' Function to create name and folder maps from a cat_to_name json map and a class_to_idx dict.
    The cat_to_name json dict contains a mapping of folder names (keys) to flower names (values)
    The class_to_idx dict contains a mapping of folder names (keys) to class indices (values)
    The class_to_idx dict keys and values are iterated over:
    If the cat_to_name dict is indexed using the class_to_idx folder names, this returns the corresponding flower name
    This flower name is then assigned as a value to the name_map dict using the class_to_idx class indices as keys, which results in a mapping of class indices to flower names
    The folder map is simply assigned the class_to_idx keys (folder names) as values, using the class_to_idx values (class indices) as keys. This results in a mapping of class indices to folder names

        Args:
            json_path (string): path of json cat_to_name file
            class_to_idx (dict): derived from DataSet.class_to_idx method

        Returns:
            name_map (dict): class index to flower name map
            folder_map (dict): class index to folder name map

    '''
    with open(json_path, 'r') as f:

        cat_to_name = json.load(f)

    name_map = {}
    folder_map = {}

    for c, i in class_to_idx.items():

        name_map[int(i)] = cat_to_name[c]
        folder_map[int(i)] = c

    return name_map, folder_map


# Predict classes function
def prediction(image_path, checkpoint, dev, k, name_map, folder_map, prints, display_results):
    ''' Function to predict the top k classes of an image using a trained neural net model.

        Args:
            image_path (string): path to image to be evaluated
            checkpoint (dict): checkpoint containing model to be evaluated
            dev (string): chosen device with which to run the model on ("cuda" or "cpu")
            k (int): number of top probabilities and classes to display
            name_map (dict): class index to flower name map
            folder_map (dict): class index to folder name map
            prints (boolean): toggle prints of top k probabilities, classes, flower names, folder names
            display_results (boolean): toggle plot of image and horizontal bar chart containing predictions

        Returns:
            probs (numpy Array): array of top k class probabilities
            classes (numpy Array): array of top k class indices
            names (list): list of top k class flower names
            folders (list): list of top k class folder names
    '''

    device = torch.device(dev)

    image = process_image(image_path)
    image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    image = image.type(torch.FloatTensor)
    image = image.to(device)

    model = checkpoint['net']['model']
    model.to(device)

    with torch.no_grad():

        model.eval()

        log_ps = model.forward(image)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(k, dim=1)
        probs = top_p.cpu().numpy()[0]
        classes = top_class.cpu().numpy()[0]
        names = []
        folders = []

        for i in classes:
            names.append(name_map[i])
            folders.append(folder_map[i])

        if prints:

            print(f"Top Probabilities: {probs}")
            print(f"Top Classes:       {classes}")
            print(f"Flower Names:      {names}")
            print(f"Folder Names:      {folders}")

        if display_results:

            im = Image.open(image_path)

            plot_prediction(im, probs, classes, name_map, folder_map)

    return probs, classes, names, folders

# Process Image function
def process_image(image_path):
    ''' Function to process an image from PIL to Tensor to be used as an input to a neural network for inference. This functions first loads a PIL Image, resizes it such that the shortest side = 255, center crops such that both sides = 224, converts PIL to numpy, squashes image such that RGB values go from range 0-255 to 0-1, Normalizes using means and stds given with the Pytorch Pretrained Model documentation, transposes such that the color channels are on axis 0, and finally converts into a torch tensor.

        Args:
            image_path (string): path to PIL image

        Returns:
            image (tensor): Should have shape of (3 x 224 x 224), with RGB values in the range of 0-1
    '''
    im = Image.open(image_path)

    # Resize to shape such that shortest side = 255
    side1, side2 = im.size

    if side1 > side2:

        im.thumbnail(((side1 / (side2 / 255)), 255))

    elif side1 < side2:

        im.thumbnail((255, (side2 / (side1 / 255))))

    else:

        im.thumbnail((255, 255))

    # Center Crop to shape 224 x 224
    side1, side2 = im.size
    new_side1 = 224
    new_side2 = 224
    left = (side1 - new_side1)/2
    right = (side1 + new_side1)/2
    top = (side2 - new_side2)/2
    bottom = (side2 + new_side2)/2
    im = im.crop((left, top, right, bottom))

    # Convert PIL to numpy and convert to float32
    image = np.array(im).astype(np.float32)

    # Squash image value range from 0-255 to 0-1
    image = image / 255

    # Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    image = (image - mean) / std

    # Reshape array from (h x w x c) to (c x h x w)
    image = image.transpose(2, 0 , 1)

    # Convert to pytorch tensor
    image = torch.from_numpy(image)

    return image


# Check predictions function
def plot_prediction(image, probs, classes, name_map, folder_map):
    ''' Function to plot a prediction of a the topk classes of an image evaluated against a trained neural network model

        Args:
            image: PIL Image
            probs (numpy Array): array of top k class probabilities
            classes (numpy Array): array of top k class indices
            names (list): list of top k class flower names
            folders (list): list of top k class folder names

        Returns:
            None
    '''

    fig, ax = plt.subplots(1, figsize=(10, 5), ncols=2)

    # show image
    ax[0].imshow(image)

    # hide image ticks and labels
    ax[0].imshow(image)
    ax[0].set_xticks([])
    ax[0].set_xticklabels([])
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])

    # bar lengths
    lengths = probs*100

    # yticks
    yticks = np.arange(len(classes))
    ax[1].set_yticks(yticks)
    ax[1].barh(yticks, lengths);
    ax[1].invert_yaxis()

    # ytick labels
    names = []

    for i in classes:
        names.append(name_map[i])

    ax[1].set_yticklabels(names);


    # xticks
    xticks = np.arange(0, 120, 20)
    z = [str(int(i)) + '%' for i in xticks]
    ax[1].set_xticks(xticks);
    ax[1].set_xticklabels(z);

    # Titles
    ax[0].set_title("Image")
    ax[1].set_title("Class Probabilities")

    fig.tight_layout()

    plt.show(block=True)

    return None



