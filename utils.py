import numpy as np
import os
import yaml
import torch



def get_arguments(parser):
    """
    Get arguments from parser.
    """
    parser.add_argument('--exp_id', type=str, default=None, help='Optional experiment identifier (3-digit number).')
    parser.add_argument('--conf_path', type=str, default='config_files/conf.yml' , help='Path to JSON config file.')
    args = parser.parse_args()
    return args

def get_config(conf_path):
    """
    Get config from yaml file.
    """
    with open(conf_path) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

def create_folder_structure(exp_id):
    """
    Create folder structure for experiment.
    prints experiment ID.
    """
    os.makedirs('experiments', exist_ok=True) # create experiments folder if it doesn't exist
    exp_folder = os.path.join('experiments', exp_id)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'best_model'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'figures'), exist_ok=True)
    print(f'Experiment ID: {exp_id}')

    return exp_folder

def get_device(verbose=True):
    """Get the device to be used for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        device_string = f'Using device: {device}'
        device_string += f'with name {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''
        print(device_string)
    return device