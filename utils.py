import numpy as np
import os
import yaml
import torch



def get_arguments(parser):
    """
    Get arguments from parser.
    """
    parser.add_argument('--exp_id', type=str, default=None, help='Optional experiment identifier (3-digit number).')
    parser.add_argument('--conf_path', type=str, default='conf.yml', help='Path to JSON config file.')
    parser.add_argument('--run_path', type=str, default=None, help='Path to config-run that contains data.')
    args = parser.parse_args()
    return args

def get_config(conf_path):
    """
    Get config from yaml file.
    """
    with open(conf_path) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

def create_exp_folder(exp_id): #! TODO: redo to be used under config file name. This is the old
    """
    Create folder structure for experiment.
    prints experiment ID.
    """
    os.makedirs('experiments', exist_ok=True) # create experiments folder if it doesn't exist
    exp_folder = os.path.join('experiments', exp_id)
    os.makedirs(exp_folder, exist_ok=True)
    # os.makedirs(os.path.join(exp_folder, 'logs'), exist_ok=True)
    # os.makedirs(os.path.join(exp_folder, 'checkpoints'), exist_ok=True)
    # os.makedirs(os.path.join(exp_folder, 'best_model'), exist_ok=True)
    # os.makedirs(os.path.join(exp_folder, 'figures'), exist_ok=True)
    print(f'Experiment ID: {exp_id}')

    return exp_folder

def create_config_dir_structure(run_dir):
    """
    Create folder structure for the specific run of a config.
    """
    # os.makedirs('experiments', exist_ok=True) # create experiments folder if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'best_model'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)


def get_device(verbose=True):
    """Get the device to be used for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        device_string = f'Using device: {device}'
        device_string += f'with name {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''
        print(device_string)
    return device