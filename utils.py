import numpy as np
import os
import json
import torch


def load_config(exp_folder):
    """Load the experiment config file"""
    with open(os.path.join(exp_folder, 'config.json')) as json_file:
        config = json.load(json_file)
    return config

def get_device(verbose=True):
    """Get the device to be used for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        device_string = f'Using device: {device}'
        device_string += f'with name {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''
        print(device_string)
    return device