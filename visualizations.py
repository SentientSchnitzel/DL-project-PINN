import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import torch

from train import Net
from utils import *

def get_exp_id(args):
    # get newest experiment id if not already given
    if args.exp_id is None:
        exp_list = os.listdir('experiments')
        exp_list.sort()
        exp_id = exp_list[-1]
    else:
        exp_id = args.exp_id
    return exp_id

def load_training_log(path):
    log_df_path = os.path.join(path, 'logs', 'training_log.csv')
    try:
        log_df = pd.read_csv(log_df_path)
        print(f'Logging data has been loaded')
        logging_plots = True
    except FileNotFoundError:
        print(f'No training log found at {log_df_path}. No logging plots will be generated.')
        # exit()
        logging_plots = False
    return log_df, logging_plots


if __name__ == '__main__':
    device = get_device(verbose=True)

    ### PARSING ###
    # parse input arguments
    parser = argparse.ArgumentParser()
    args = get_arguments(parser)

    run_path = args.run_path

    run_conf = get_config(os.path.join(run_path, 'run_config.yml')) if args.conf_path is None else get_config(args.conf_path)
    figures_folder = os.path.join(run_path, 'figures')

    ### yaml arguments
    n_logs = run_conf['logging']['n_logs']
    epochs = run_conf['training']['epochs']

    log_df, logging_plots = load_training_log(run_path)

    ### static plot parameters
    plt.rcParams['font.size'] = 12
    quality = 200 #dpi
    size = (10, 6) #inches

    if logging_plots:
        # Calculate the logging interval and create x_ticks
        log_interval = epochs // n_logs
        x_ticks = np.arange(0, epochs, step=log_interval)

        tick_spacing = 400  # Define the spacing between each x-tick #! TODO relate the spacing to the number of epochs logged (dynamic)
        custom_ticks = np.arange(0, epochs + 1, tick_spacing)
        custom_labels = [str(int(tick)) for tick in custom_ticks]

        # Get the index of the minimum loss
        min_loss_index = log_df['Total Loss'].idxmin()
        min_loss_value = log_df['Total Loss'][min_loss_index]
        min_loss_epoch = x_ticks[min_loss_index]

        fig, ax = plt.subplots(figsize=(10, 10), nrows=4, dpi=200)

        # Plot Total Loss
        ax[0].plot(x_ticks, log_df['Total Loss'], label='Total Loss')
        # Annotate the minimum loss
        text_offset = 100
        ax[0].annotate(f'min loss:\n{min_loss_value:.5f}',
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch - text_offset, min_loss_value*100),
                #    arrowprops=dict(facecolor='black', shrink=1, width=1, headwidth=3),
                horizontalalignment='center',
                verticalalignment='center')

        # Draw a vertical line at the index of minimum loss
        ax[0].axvline(x=min_loss_epoch, color='red', linestyle='--', linewidth=1)

        ax[0].set_xticks(custom_ticks)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_yscale('log')
        ax[0].set_title('Total Loss over Epochs')
        ax[0].legend()

        # Plot Individual Loss Terms
        ax[1].plot(x_ticks, log_df['PDE Loss'], label='PDE Loss')
        ax[1].plot(x_ticks, log_df['IC1 Loss'], label='IC1 Loss')
        ax[1].plot(x_ticks, log_df['IC2 Loss'], label='IC2 Loss')
        ax[1].plot(x_ticks, log_df['BC Loss'], label='BC Loss')
        ax[1].set_xticks(custom_ticks)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].set_yscale('log')
        ax[1].set_title('Individual Loss Terms over Epochs')
        ax[1].legend()

        # Plot Weights
        ax[2].plot(x_ticks, log_df['lambda PDE'], label='lambda PDE')
        ax[2].plot(x_ticks, log_df['lambda IC1'], label='lambda IC1')
        ax[2].plot(x_ticks, log_df['lambda IC2'], label='lambda IC2')
        ax[2].plot(x_ticks, log_df['lambda BC'], label='lambda BC')
        ax[2].set_xticks(custom_ticks)
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('Weight')
        ax[2].set_title('Weights Applied to Loss Terms over Epochs')
        ax[2].legend()

        # Plot Learning Rate
        ax[3].plot(x_ticks, log_df['Learning Rate'], label='Learning Rate')
        ax[3].set_xticks(custom_ticks)
        ax[3].set_xlabel('Epochs')
        ax[3].set_ylabel('Learning Rate')
        ax[3].set_title('Learning Rate across Epochs')
        ax[3].legend()

        plt.tight_layout()
        plt.show()

        fig.savefig(os.path.join(figures_folder, 'training_log.png'))
        print(f'Logging figure has been saved')


    # Define your model architecture (must match the saved model)
    model = Net(input_dim=2, 
                output_dim=1, 
                hidden_dim=run_conf['training']['hidden_dim'], 
                n_layers=run_conf['training']['n_layers'],
                dropout_rate=run_conf['training']['dropout_rate'],
                initialization=run_conf['training']['initialization']
                )  # Replace with your actual model architecture

    # Load the best model (or a specific checkpoint)
    try:
        # get the best model
        best_model_path = os.path.join(run_path, 'best_model', 'best_model.pt')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f'Best model has been loaded')
    except FileNotFoundError:
        print(f'No best model found at best_model/best_model.pt.')
        # get the latest checkpoint
        checkpoint_folder = os.path.join(run_path, 'checkpoints')
        checkpoints = os.listdir(checkpoint_folder)
        checkpoints.sort()
        checkpoint_path = os.path.join(checkpoint_folder, checkpoints[-1])
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Checkpoint {checkpoint_path} has been loaded instead.')


    ### Finite Difference Ground Truth Solution
    def calc_ground_truth(x_domain, t_domain, c, sigma, source_point):
        
        Lx = x_domain[1] - x_domain[0]  # Length of the domain
        Lt = t_domain[1] - t_domain[0]  # Length of the time domain
        Nx = int(Lx * 500)
        Nt = int(Lt * 500)
        dx = Lx / Nx
        dt = Lt / Nt    # Smaller time step for stability

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

        x_space = torch.linspace(*x_domain, Nx)
        t_space = torch.linspace(*t_domain, Nt)
        return u, x_space, t_space

    ### Finite Difference Ground Truth Solution
    x_domain = run_conf['data']['x_domain']
    t_domain = run_conf['data']['t_domain']
    t_domain_ext = [ t_domain[0], 2 * t_domain[1] ] # extend the time domain for use in extrapolation
    c = run_conf['physics']['wave_speed']
    sigma = run_conf['physics']['sigma']
    source_point = run_conf['physics']['x0']

    u, x_space, t_space = calc_ground_truth(x_domain, t_domain_ext, c, sigma, source_point)

    # Plot the ground truth solution
    plt.figure(figsize=size, dpi=quality)
    plt.contourf(x_space, t_space, u, levels=100, cmap='viridis')
    plt.title('Wave Propagation')
    plt.axhline(y=t_domain[1], color='red', linestyle='--', linewidth=1) #t_domain[1]
    plt.xlabel('Space')
    plt.ylabel('Time')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 'ground_truth.png'))
    plt.show()
    print(f'Ground truth figure has been saved')


    # Generate predictions over time and space
    X_mesh, T_mesh = torch.meshgrid(x_space, t_space, indexing='ij')
    feature_grid = torch.stack((X_mesh.flatten(), T_mesh.flatten()), dim=1)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(feature_grid).reshape(X_mesh.shape)
    
    print(t_space.shape)
    print(X_mesh.shape)
    print(predictions.shape)
    print(u.shape)

    # Plot Model Predictions
    plt.figure(figsize=size, dpi=quality)
    plt.contourf(X_mesh.numpy(), T_mesh.numpy(), predictions.numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='Model Prediction')
    plt.axhline(y=t_domain[1], color='red', linestyle='--', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Model Predictions Over Time and Space')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 'model_predictions.png'))
    plt.show()
    print(f'Prediction figure has been saved')


    # plot the ground truth vs the model predictions in same figure
    min_value = min(u.min(), predictions.numpy().min())
    max_value = max(u.max(), predictions.numpy().max())
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=quality, constrained_layout=True)

    # Plot Ground Truth
    contour_gt = ax[0].contourf(x_space, t_space, u, levels=100, cmap='viridis', vmin=min_value, vmax=max_value)
    ax[0].set_title('Ground Truth')
    ax[0].axhline(y=t_domain[1], color='red', linestyle='--', linewidth=2)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    # Plot Model Predictions
    contour_pred = ax[1].contourf(X_mesh.numpy(), T_mesh.numpy(), predictions.numpy(), levels=100, cmap='viridis', vmin=min_value, vmax=max_value)
    ax[1].set_title('Model Predictions')
    ax[1].axhline(y=t_domain[1], color='red', linestyle='--', linewidth=2)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    # Create Colorbar for Both Axes
    fig.colorbar(contour_pred, ax=ax, fraction=0.1, label='Amplitude')

    # Save and Show the Plot
    plt.savefig(os.path.join(figures_folder, 'ground_truth_vs_model_predictions.png'))
    plt.show()
    print(f'Ground truth vs model predictions figure has been saved')

    
    # calculate the MSE between the ground truth and the model predictions
    # in-domain
    mse = torch.nn.MSELoss()
    mse_in = mse(torch.from_numpy(u.T[:, :1000]), predictions[:, :1000])
    # out-of-domain
    mse_out = mse(torch.from_numpy(u.T[:, 1000:]), predictions[:, 1000:])
    print(f'MSE between ground truth and model predictions: \nIn-domain {mse_in:.5f} \nOut-of-domain: {mse_out:.5f}')

    
    ### Plot the difference between the ground truth and the model predictions
    plt.figure(figsize=size, dpi=quality)
    diff = u.T - predictions.numpy() # rotate the predictions to match the ground truth
    # plt.contourf(x_space, t_space, diff, levels=100, cmap='viridis')
    plt.contourf(X_mesh.numpy(), T_mesh.numpy(), diff, levels=100, cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.axhline(y=t_domain[1], color='red', linestyle='--', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Difference Between Ground Truth and Model Predictions')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 'difference.png'))
    plt.show()
    print(f'Difference figure has been saved')



    # load all 10 model checkpoints and test their predictions
    # at t=0 for the entire x_domain
    checkpoint_folder = os.path.join(run_path, 'checkpoints')
    checkpoint_list = os.listdir(checkpoint_folder)

    # initialize the predictions array
    predictions = np.zeros((len(checkpoint_list), len(x_space)))

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
    # plot ground truth at t=0 for 10 checkpoints to see evolution
    plt.plot(x_space.numpy(), u[0, :], label='Ground Truth', linestyle='--', color='black', linewidth=3)
    for i in range(len(checkpoint_list)):
        plt.plot(x_space.numpy(), predictions[i, :], alpha=0.5, label=f'Checkpoint {i+1}')

    # best model prediction
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        best_model_predictions = model(torch.stack((x_space, torch.zeros_like(x_space)), dim=1)).flatten().numpy()
    plt.plot(x_space.numpy(), best_model_predictions, label='Best Model', color='red', linewidth=3)

    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.title('Gaussian source predictions at t=0')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 't0_gaussian_multiple_predictions.png'))
    plt.show()
    print(f't=0 Gaussian source predictions figure has been saved')

    plt.figure(figsize=size, dpi=quality)
    plt.plot(x_space.numpy(), u[0, :], label='Ground Truth', linestyle='--', color='black', linewidth=3)
    plt.plot(x_space.numpy(), best_model_predictions, label='Best Model', color='red', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.title('Gaussian source prediction at t=0')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 't0_gaussian_prediction.png'))
    plt.show()
    print(f't=0 Gaussian source prediction figure has been saved')
