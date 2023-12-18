import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

import torch

from wave_eq_pinn_minibatching import Net  # Replace with your actual model class
from utils import *


### PARSING ###
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID (default: newest)')
args = parser.parse_args()

# get newest experiment id
if args.exp_id is None:
    exp_list = os.listdir('experiments')
    exp_id = max(exp_list)
else:
    exp_id = args.exp_id

# get device
device = get_device(verbose=True)


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

if logging_plots:
    fig, ax = plt.subplots(figsize=(10, 10), nrows=3, dpi=200)

    # Plot Total Loss
    ax[0].plot(log_df['Total Loss'], label='Total Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_yscale('log')
    ax[0].set_title('Total Loss over Epochs')
    ax[0].legend()

    # Plot Individual Loss Terms
    ax[1].plot(log_df['PDE Loss'], label='PDE Loss')
    ax[1].plot(log_df['IC1 Loss'], label='IC1 Loss')
    ax[1].plot(log_df['IC2 Loss'], label='IC2 Loss')
    ax[1].plot(log_df['BC Loss'], label='BC Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    ax[1].set_title('Individual Loss Terms over Epochs')
    ax[1].legend()

    # Plot Weights
    ax[2].plot(log_df['lambda PDE'], label='lambda PDE')
    ax[2].plot(log_df['lambda IC1'], label='lambda IC1')
    ax[2].plot(log_df['lambda IC2'], label='lambda IC2')
    ax[2].plot(log_df['lambda BC'], label='lambda BC')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Weight')
    ax[2].set_title('Weights Applied to Loss Terms over Epochs')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

    # save figure
    fig.savefig(os.path.join(exp_folder, 'logs', 'training_log.png'))
    print(f'Logging figure has been saved')

# Define your model architecture (must match the saved model)
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

def calc_ground_truth(x_domain, t_domain):
    Lx = x_domain[1] - x_domain[0]  # Length of the domain
    Lt  = t_domain[1] - t_domain[0]  # Length of the time domain
    Nx = Lx * 500        # Number of spatial points
    Nt = Lt * 500       # Number of time points
    # print(f'Nx: {Nx}, Nt: {Nt}')
    c = 1           # Wave speed
    dx = Lx / Nx
    dt = Lt / Nt    # Smaller time step for stability
    # alpha = 50      # Gaussian width parameter
    sigma = 0.3     # Gaussian width parameter
    source_point = 0.0  # Position of the Gaussian source

    # Spatial grid
    x = np.linspace(*x_domain, Nx)

    # intitialize the wave field
    u = np.zeros((Nt, Nx))
    u[0, :] = np.exp(-((x - source_point)/sigma)**2)

    u[1, :] = u[0, :]  # No initial velocity

    # Finite difference method
    for n in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            u[n + 1, i] = 2*u[n, i] - u[n - 1, i] + (dt**2 * c**2) * (u[n, i - 1] - 2*u[n, i] + u[n, i + 1]) / dx**2

        # Neumann boundary conditions
        u[n + 1, 0] = u[n + 1, 1]
        u[n + 1, -1] = u[n + 1, -2]

    u = np.flip(u, axis=0)  # flip the array to match the model predictions
    x_space = torch.linspace(*x_domain, Nx)
    t_space = torch.linspace(*t_domain, Nt)
    return u,x_space,t_space

### Finite Difference Ground Truth Solution
x_domain = (-1, 1)
t_domain = (0, 2)  # Define T domain

u,x_space,t_space = calc_ground_truth(x_domain, t_domain)

# Plot the ground truth solution
plt.figure(figsize=size, dpi=quality)
plt.imshow(u, extent=[*x_domain, *t_domain], aspect='auto', cmap='viridis')
plt.title('Wave Propagation')
plt.xlabel('Space')
plt.ylabel('Time')
plt.colorbar(label='Amplitude')
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'ground_truth.png'))
print(f'Ground truth figure has been saved')



# Generate predictions over time and space
X_mesh, T_mesh = torch.meshgrid(x_space, t_space, indexing='ij')
feature_grid = torch.stack((X_mesh.flatten(), T_mesh.flatten()), dim=1)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(feature_grid).reshape(X_mesh.shape)

# Plot Model Predictions
plt.figure(figsize=size, dpi=quality)
plt.contourf(X_mesh.numpy(), T_mesh.numpy(), predictions.numpy(), levels=100, cmap='viridis')
plt.colorbar(label='Model Prediction')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Model Predictions Over Time and Space')
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'model_predictions.png'))
print(f'Prediction figure has been saved')


plt.figure(figsize=size, dpi=quality)
# plot ground truth at t=0
plt.plot(x_space.numpy(), u[0, :], label='Ground Truth', color='orange', linewidth=3)
# Plot the predictions
plt.plot(x_space.numpy(), predictions[0,:], label='best model', linewidth=3)
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title('Gaussian source predictions at t=0')
plt.legend()
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'best_model_predictions.png'))
print(f'best model prediction figure has been saved')



# Plot Model Predictions and ground truth solution in subplots
fig, ax = plt.subplots(figsize=(10, 5), ncols=2, dpi=200)

# Plot Model Predictions
contour = ax[1].contourf(X_mesh.numpy(), T_mesh.numpy(), predictions.numpy(), levels=100, cmap='viridis')
ax[1].set_xlabel('x')
ax[1].set_ylabel('t')
ax[1].set_title('Model Predictions')

# Plot the ground truth solution
image = ax[0].imshow(u, extent=[*x_domain, *t_domain], aspect='auto', cmap='viridis')
ax[0].set_title('Ground Truth')
ax[0].set_xlabel('x')

# Create a common colorbar for both subplots
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(contour, cax=cax, orientation='vertical')
cbar.set_label('Amplitude')  # You can customize the label as needed

# Adjust layout
plt.subplots_adjust(right=0.9)  # Adjust the right margin for the colorbar

# Show the plot
plt.show()


# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'combined.png'))
print(f'combined Ground truth and Prediction figure figure has been saved')



### Plot the interpolation error
plt.figure(figsize=size, dpi=quality)
# Plot the predictions 

# Generate interpolation prediction
t_domain_inter = (0, 4)
u_interpolation,x_space_inter,t_space_inter = calc_ground_truth(x_domain, t_domain_inter)
# Plot the interpolation ground truth solution
plt.figure(figsize=size, dpi=quality)

plt.imshow(u_interpolation, extent=[*x_domain, *t_domain_inter], aspect='auto', cmap='viridis')
## add a  line to show the interpolation time
plt.plot(x_space_inter, np.ones_like(x_space_inter)*2, color='red', linewidth=2, linestyle='--')
plt.title('Wave Propagation')
plt.xlabel('Space')
plt.ylabel('Time')
plt.colorbar(label='Amplitude')
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'ground_truth_extrapolation.png'))
print(f'Ground truth extrapolation figure has been saved')

# t_space_inter = torch.linspace(*t_domain_inter, 1500)
X_mesh_inter, T_mesh_inter = torch.meshgrid(x_space_inter, t_space_inter, indexing='ij')
feature_grid_inter = torch.stack((X_mesh_inter.flatten(), T_mesh_inter.flatten()), dim=1)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions_inter = model(feature_grid_inter).reshape(X_mesh_inter.shape)

plt.contourf(X_mesh_inter.numpy(), T_mesh_inter.numpy(), predictions_inter.numpy(), levels=100, cmap='viridis')
plt.colorbar(label='Model extrapolation')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Model extrapolation Over Time and Space')
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'model_prediction_extrapolation.png'))
print(f'prediction extrapolation saved')


# Plot Model Predictions and ground truth solution in subplots
fig, ax = plt.subplots(figsize=(10, 5), ncols=2, dpi=200)

# Plot Model Predictionsinterpolation
contour = ax[1].contourf(X_mesh_inter.numpy(), T_mesh_inter.numpy(), predictions_inter.numpy(), levels=100, cmap='viridis')
ax[1].plot(x_space_inter, np.ones_like(x_space_inter)*2, color='red', linewidth=2, linestyle='--')
ax[1].set_xlabel('x')
ax[1].set_ylabel('t')
ax[1].set_title('Model Predictions')

# Plot the ground truth solution
image = ax[0].imshow(u_interpolation, extent=[*x_domain, *t_domain_inter], aspect='auto', cmap='viridis')
ax[0].plot(x_space_inter, np.ones_like(x_space_inter)*2, color='red', linewidth=2, linestyle='--')
ax[0].set_title('Ground Truth')
ax[0].set_xlabel('x')

# Create a common colorbar for both subplots
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(contour, cax=cax, orientation='vertical')
cbar.set_label('Amplitude')  # You can customize the label as needed

# Adjust layout
plt.subplots_adjust(right=0.9)  # Adjust the right margin for the colorbar

# Show the plot
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'combined_extrapolation.png'))
print(f'combined Ground truth and extrapolation figure has been saved')

### Plot the difference between the ground truth and the model predictions
plt.figure(figsize=size, dpi=quality)
diff = u - predictions.rot90(1).numpy() # rotate and flip the predictions to match the ground truth
print(f'Maximum difference: {np.max(np.abs(diff))}')
plt.imshow(diff, extent=[*x_domain, *t_domain], aspect='auto', cmap='viridis')
# plt.contourf(X_mesh.numpy(), T_mesh.numpy(), u - predictions.numpy(), levels=100, cmap='viridis')
plt.colorbar(label='Amplitude')
plt.clim(min(np.min(diff),0), 0.2)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Difference Between Ground Truth and Model Predictions')
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'difference.png'))
print(f'Difference figure has been saved')


### Plot the difference between the ground truth and the model predictions
plt.figure(figsize=size, dpi=quality)
diff_inter = u_interpolation - predictions_inter.rot90(1).numpy() # rotate and flip the predictions to match the ground truth
print(f'Maximum difference extrapolation: {np.max(np.abs(diff_inter))}')
plt.imshow(diff_inter, extent=[*x_domain, *t_domain_inter], aspect='auto', cmap='viridis')
# plt.contourf(X_mesh.numpy(), T_mesh.numpy(), u - predictions.numpy(), levels=100, cmap='viridis')
plt.plot(x_space_inter, np.ones_like(x_space_inter)*2, color='red', linewidth=2, linestyle='--')
plt.colorbar(label='Amplitude')
plt.clim(min(np.min(diff_inter),0), max(np.max(diff_inter),0.2))
plt.xlabel('x')
plt.ylabel('t')
plt.title('Difference Between Ground Truth and Model extrapolation')
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'difference_extrapolation.png'))
print(f'Difference extrapolation figure has been saved')


# load all 10 model checkpoints and test their predictions
# at t=0 for the entire x_domain
checkpoint_folder = os.path.join(exp_folder, 'checkpoints')
checkpoint_list = os.listdir(checkpoint_folder)

# initialize the predictions array
predictions = np.zeros((len(checkpoint_list), 1000))

# loop through all checkpoints
for i, checkpoint in enumerate(checkpoint_list):
    # load the model
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # generate predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions[i, :] = model(torch.stack((x_space, torch.zeros_like(x_space)), dim=1)).flatten().numpy()


plt.figure(figsize=size, dpi=quality)
# plot ground truth at t=0
plt.plot(x_space.numpy(), u[0, :], label='Ground Truth', linestyle='--', color='black', linewidth=3)
# Plot the predictions
for i in range(len(checkpoint_list)):
    plt.plot(x_space.numpy(), predictions[i, :], label=f'Checkpoint {i+1}')

plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title('Gaussian source predictions at t=0')
plt.legend()
plt.show()

# save the figure in experiments logs folder
plt.savefig(os.path.join(exp_folder, 'logs', 'checkpoint_predictions.png'))
print(f'Checkpoint prediction figure has been saved')



