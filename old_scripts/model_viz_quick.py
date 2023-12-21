#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

import torch

from train import Net
from utils import *


device = get_device(verbose=True)

# get newest experiment id
if exp_id is None:
    exp_list = os.listdir('experiments')
    exp_list.sort()
    exp_id = exp_list[-1]
else:
    exp_id = exp_id

exp_id = '087'
exp_folder = create_folder_structure(exp_id)
conf = get_config(os.path.join(exp_folder, 'conf.yml')) # get the config file from the experiment folder

n_logs = conf['logging']['n_logs']
epochs = conf['training']['epochs']




# Load the training log
exp_folder = os.path.join('experiments', exp_id)
print(f'Loading experiment from {exp_folder}')
log_df_path = os.path.join(exp_folder, 'logs', 'training_log.csv')
try:
    log_df = pd.read_csv(log_df_path)
    print(f'Logging data has been loaded')
    logging_plots = True
except FileNotFoundError:
    print(f'No training log found at {log_df_path}.')
    # exit()
    logging_plots = False


### static plot parameters
plt.rcParams['font.size'] = 12
quality = 200 #dpi
size = (10, 6) #inches


# Define your model architecture (must match the saved model)
#! TODO: replace with loading of JSON file from experiment folder.
model = Net(input_dim=2, output_dim=1, hidden_dim=32, n_layers=2)  # Replace with your actual model architecture

# Load the best model (or a specific checkpoint)
try:
    # get the best model
    best_model_path = os.path.join(exp_folder, 'best_model', 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f'Best model has been loaded')
except FileNotFoundError:
    print(f'No best model found at best_model/best_model.pt.')
    # get the latest checkpoint
    checkpoint_folder = os.path.join(exp_folder, 'checkpoints')
    checkpoints = os.listdir(checkpoint_folder)
    checkpoints.sort()
    checkpoint_path = os.path.join(checkpoint_folder, checkpoints[-1])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f'Checkpoint {checkpoint_path} has been loaded instead.')

# f

#%%


#%%
# plot all the model's layers' weights in separate plots

# get the weights
weights_x = []
weights_t = []
for param in model.parameters():
    print(param.shape)
#%%

weights_x[0]