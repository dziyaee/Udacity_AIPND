# Imports here

# General imports
import argparse

# Pytorch imports
import torch

# testing / inference function imports
from functions_testing import load_checkpoint, json_maps, prediction, process_image, plot_prediction

#---------------------------------------------------------------------
print("-" * 100)

parser = argparse.ArgumentParser()

parser.add_argument('image_path',
                    action = 'store',
                    type = str,
                    help = 'path of the image for the network to make a prediction on'
                   )

parser.add_argument('checkpoint_path',
                    action = 'store',
                    type = str,
                    help = 'path of the checkpoint to load the network'
                   )

parser.add_argument('--top_k',
                    action = 'store',
                    default = 5,
                    type = int,
                    help = 'Number of top classes to display, default = 5'
                   )

parser.add_argument('--json_path',
                    action = 'store',
                    default = 'cat_to_name.json',
                    type = str,
                    help = 'json file containing category to name mapping, default = "cat_to_name.json"'
                   )

parser.add_argument('--gpu',
                    action = 'store_true',
                    help = 'use the gpu for training, default = False'
                   )

args = parser.parse_args()

# Keep topk between 1 and 10 to prevent histogram from being overcrowded with results
if (args.top_k <= 0) or (args.top_k > 10):
    parser.error("top_k must be between 1 and 10")

# Variable assignment
image_path = args.image_path
checkpoint_path = args.checkpoint_path
top_k = args.top_k
json_path = args.json_path
gpu = args.gpu

# Device assignment
if gpu:

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

# Prints
print("\n")
print(f"image_path    = {image_path}")
print(f"checkpoint_path = {checkpoint_path}")
print(f"top_k           = {top_k}")
print(f"json_path       = {json_path}")
print(f"GPU             = {gpu}")
print("\n")
print("-" * 100)

# Print and display prediction results
prints = True
display_results = True

# Load Checkpoint
checkpoint = load_checkpoint(checkpoint_path)

# Get folder to index map
class_to_idx = checkpoint['class_to_idx']

# Get index to name and index to folder maps
name_map, folder_map = json_maps(json_path, class_to_idx)

# Run prediction on image
prediction(image_path, checkpoint, device, top_k, name_map, folder_map, prints, display_results)


