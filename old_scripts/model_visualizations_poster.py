import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

import torch

from train import Net  # Replace with your actual model class
from utils import *


### PARSING ###
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID (default: newest)')
args = parser.parse_args()

### JSON arguments
n_logs = 100
epochs = 2000

exp_dir = 'experiments'

# get newest experiment id
if args.exp_id is None:
    exp_list = os.listdir(exp_dir)
    exp_list.sort()
    exp_id = exp_list[-1]
else:
    exp_id = args.exp_id

# get device
device = get_device(verbose=True)

def load_logs(base_dir, exp_id):
    # Load the training log
    exp_folder = os.path.join(base_dir, exp_id)
    print(f'Loading experiment from {exp_folder}')
    log_df_path = os.path.join(exp_folder, 'logs', 'training_log.csv')
    try:
        log_df = pd.read_csv(log_df_path)
        print(f'Logging data has been loaded')
        logging_plots = True
    except FileNotFoundError:
        print(f'No training log found at {log_df_path}.')
        logging_plots = False

    return log_df, logging_plots, exp_folder


# # Load the training log
# exp_folder = os.path.join('experiments', exp_id)
# print(f'Loading experiment from {exp_folder}')
# log_df_path = os.path.join(exp_folder, 'logs', 'training_log.csv')
# try:
#     log_df = pd.read_csv(log_df_path)
#     print(f'Logging data has been loaded')
#     logging_plots = True
# except FileNotFoundError:
#     print(f'No training log found at {log_df_path}.')
#     # exit()
#     logging_plots = False


# loading logs for all experiments
# log_df_static, _ = load_logs('experiments', exp_id='088')
log_df, logging_plots, exp_folder = load_logs(exp_dir, exp_id=exp_id)



### static plot parameters
plt.rcParams['font.size'] = 12
quality = 300 #dpi
size = (10, 6) #inches

if logging_plots:
    # Calculate the logging interval and create x_ticks
    log_interval = epochs // n_logs
    x_ticks = np.arange(0, epochs, step=log_interval)

    tick_spacing = 400  # Define the spacing between each x-tick
    custom_ticks = np.arange(0, epochs + 1, tick_spacing)
    custom_labels = [str(int(tick)) for tick in custom_ticks]

    # Get the index of the minimum loss
    min_loss_index = log_df['Total Loss'].idxmin()
    min_loss_value = log_df['Total Loss'][min_loss_index]
    min_loss_epoch = x_ticks[min_loss_index]

    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, dpi=200)

    # # Plot Total Loss
    # ax[0].plot(x_ticks, log_df['Total Loss'], label='Total Loss')
    # ax[0].set_xticks(custom_ticks)
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('Loss')
    # ax[0].set_yscale('log')
    # ax[0].set_title('Total Loss over Epochs')
    # ax[0].legend()

    # Plot Individual Loss Terms
    ax[0].plot(x_ticks, log_df['PDE Loss'], label='PDE Loss')
    ax[0].plot(x_ticks, log_df['IC1 Loss'], label='IC1 Loss')
    ax[0].plot(x_ticks, log_df['IC2 Loss'], label='IC2 Loss')
    ax[0].plot(x_ticks, log_df['BC Loss'], label='BC Loss')

    # Annotate the minimum loss
    ax[0].annotate(f'min loss:\n{min_loss_value:.5f}',
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch, min_loss_value*2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                horizontalalignment='center')

    # Draw a vertical line at the index of minimum loss
    ax[0].axvline(x=min_loss_epoch, color='red', linestyle='--', linewidth=1)

    ax[0].set_xticks(custom_ticks)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_yscale('log')
    ax[0].set_title('Individual Loss Terms over Epochs')
    ax[0].legend()

    # Plot Weights
    ax[1].plot(x_ticks, log_df['lambda PDE'], label='lambda PDE')
    ax[1].plot(x_ticks, log_df['lambda IC1'], label='lambda IC1')
    ax[1].plot(x_ticks, log_df['lambda IC2'], label='lambda IC2')
    ax[1].plot(x_ticks, log_df['lambda BC'], label='lambda BC')
    ax[1].set_xticks(custom_ticks)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Weight')
    ax[1].set_title('Weights Applied to Loss Terms over Epochs')
    ax[1].legend()

    # # Plot Learning Rate
    # ax[3].plot(x_ticks, log_df['Learning Rate'], label='Learning Rate')
    # ax[3].set_xticks(custom_ticks)
    # ax[3].set_xlabel('Epochs')
    # ax[3].set_ylabel('Learning Rate')
    # ax[3].set_title('Learning Rate across Epochs')
    # ax[3].legend()

    plt.tight_layout()
    plt.show()

    # save figure
    fig.savefig(os.path.join(exp_folder, 'logs', 'training_log.png'))
    print(f'Logging figure has been saved')


# # Define your model architecture (must match the saved model)
# #! TODO: replace with loading of JSON file from experiment folder.
# model = Net(input_dim=2, output_dim=1, hidden_dim=32, n_layers=2)  # Replace with your actual model architecture

# # Load the best model (or a specific checkpoint)
# try:
#     # get the best model
#     best_model_path = os.path.join(exp_folder, 'best_model', 'best_model.pt')
#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     print(f'Best model has been loaded')
# except FileNotFoundError:
#     print(f'No best model found at best_model/best_model.pt.')
#     # get the latest checkpoint
#     checkpoint_folder = os.path.join(exp_folder, 'checkpoints')
#     checkpoints = os.listdir(checkpoint_folder)
#     checkpoints.sort()
#     checkpoint_path = os.path.join(checkpoint_folder, checkpoints[-1])
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     print(f'Checkpoint {checkpoint_path} has been loaded instead.')



# ### Finite Difference Ground Truth Solution
# x_domain = (-1, 1)
# t_domain = (0, 2)  # Define T domain

# Lx = x_domain[1] - x_domain[0]  # Length of the domain
# Lt  = t_domain[1] - t_domain[0]  # Length of the time domain
# Nx = 1000        # Number of spatial points
# Nt = 1000        # Number of time points
# c = 1           # Wave speed
# dx = Lx / Nx
# dt = Lt / Nt    # Smaller time step for stability
# alpha = 50      # Gaussian width parameter
# source_point = 0.0  # Position of the Gaussian source

# # Spatial grid
# x = np.linspace(*x_domain, Nx)

# # intitialize the wave field
# u = np.zeros((Nt, Nx))
# u[0, :] = np.exp(-alpha * (x - source_point)**2)

# u[1, :] = u[0, :]  # No initial velocity

# # Finite difference method
# for n in range(1, Nt - 1):
#     for i in range(1, Nx - 1):
#         u[n + 1, i] = 2*u[n, i] - u[n - 1, i] + (dt**2 * c**2) * (u[n, i - 1] - 2*u[n, i] + u[n, i + 1]) / dx**2

#     # Neumann boundary conditions
#     u[n + 1, 0] = u[n + 1, 1]
#     u[n + 1, -1] = u[n + 1, -2]

# # Plot the ground truth solution
# plt.figure(figsize=size, dpi=quality)
# plt.imshow(u, extent=[*x_domain, *t_domain], aspect='auto', cmap='viridis')
# plt.title('Wave Propagation')
# plt.xlabel('Space')
# plt.ylabel('Time')
# plt.colorbar(label='Amplitude')
# plt.show()

# # save the figure in experiments logs folder
# plt.savefig(os.path.join(exp_folder, 'logs', 'ground_truth.png'))
# print(f'Ground truth figure has been saved')



# # Generate predictions over time and space
# x_space = torch.linspace(*x_domain, Nx)
# t_space = torch.linspace(*t_domain, Nt)
# X_mesh, T_mesh = torch.meshgrid(x_space, t_space, indexing='ij')
# feature_grid = torch.stack((X_mesh.flatten(), T_mesh.flatten()), dim=1)

# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     predictions = model(feature_grid).reshape(X_mesh.shape)

# # Plot Model Predictions
# plt.figure(figsize=size, dpi=quality)
# plt.contourf(X_mesh.numpy(), T_mesh.numpy(), predictions.numpy(), levels=100, cmap='viridis')
# plt.colorbar(label='Model Prediction')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title('Model Predictions Over Time and Space')
# plt.show()

# # save the figure in experiments logs folder
# plt.savefig(os.path.join(exp_folder, 'logs', 'model_predictions.png'))
# print(f'Prediction figure has been saved')



# ### Plot the difference between the ground truth and the model predictions
# plt.figure(figsize=size, dpi=quality)
# diff = u - predictions.rot90(1).numpy() # rotate and flip the predictions to match the ground truth
# plt.imshow(diff, extent=[*x_domain, *t_domain], aspect='auto', cmap='viridis')
# # plt.contourf(X_mesh.numpy(), T_mesh.numpy(), u - predictions.numpy(), levels=100, cmap='viridis')
# plt.colorbar(label='Amplitude')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title('Difference Between Ground Truth and Model Predictions')
# plt.show()

# # save the figure in experiments logs folder
# plt.savefig(os.path.join(exp_folder, 'logs', 'difference.png'))
# print(f'Difference figure has been saved')



# # load all 10 model checkpoints and test their predictions
# # at t=0 for the entire x_domain
# checkpoint_folder = os.path.join(exp_folder, 'checkpoints')
# checkpoint_list = os.listdir(checkpoint_folder)

# # initialize the predictions array
# predictions = np.zeros((len(checkpoint_list), Nx))

# # loop through all checkpoints
# for i, checkpoint in enumerate(checkpoint_list):
#     # load the model
#     checkpoint_path = os.path.join(checkpoint_folder, checkpoint)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))

#     # generate predictions
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         predictions[i, :] = model(torch.stack((x_space, torch.zeros_like(x_space)), dim=1)).flatten().numpy()


# plt.figure(figsize=size, dpi=quality)
# # plot ground truth at t=0
# plt.plot(x_space.numpy(), u[0, :], label='Ground Truth', linestyle='--', color='black', linewidth=3)
# # Plot the predictions
# for i in range(len(checkpoint_list)):
#     plt.plot(x_space.numpy(), predictions[i, :], label=f'Checkpoint {i+1}')

# plt.xlabel('x')
# plt.ylabel('Amplitude')
# plt.title('Gaussian source predictions at t=0')
# plt.legend()
# plt.show()

# # save the figure in experiments logs folder
# plt.savefig(os.path.join(exp_folder, 'logs', 'checkpoint_predictions.png'))
# print(f'Checkpoint prediction figure has been saved')