import numpy as np
import pandas as pd
import os
import argparse
import re

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
#from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR

from tqdm import tqdm



def next_experiment_number(base_dir):
    """Finds the next experiment number based on existing directories."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return '001'

    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    existing_nums = [int(re.search(r'\d+', d).group()) for d in existing_dirs if re.search(r'\d+', d)]

    if not existing_nums:
        return '001'
    max_num = max(existing_nums)
    next_num = str(max_num + 1).zfill(3)
    return next_num


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):

        super(Net, self).__init__()

        activation = nn.Softplus # nn.LeakyReLU, nn.Tanh, nn.ReLU, nn.ELU

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation()
        )

        self.network = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                # nn.BatchNorm1d(hidden_dim),
                # nn.Dropout(0.3),
            ) for i in range(n_layers)]
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.network(x)
        x = self.output(x)
        return x


def calc_ground_truth(x_domain, t_domain):
    resolution = 16
    Lx = x_domain[1] - x_domain[0]  # Length of the domain
    Lt  = t_domain[1] - t_domain[0]  # Length of the time domain
    Nx = Lx * resolution        # Number of spatial points
    Nt = Lt * resolution       # Number of time points
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

    # u = np.flip(u, axis=0)  # flip the array to match the model predictions
    x_space = torch.linspace(*x_domain, Nx)
    t_space = torch.linspace(*t_domain, Nt)
    return u,x_space,t_space

def train(ffn, optimizer, feature_loader, targets_loader, criterion, boundaries, v, epochs, exp_folder, device, scheduler):
    """
    ffn:        FFN model
    criterion:  loss function
    optimizer:  optimizer
    dataloader: torch dataloader
    boundaries: tuple of boundary values
    v:          wave speed
    epochs:     number of epochs
    save_path:  path to save the model
    """

    assert len(feature_loader) == len(targets_loader), 'feature_loader and targets_loader must have the same length'
    assert feature_loader.batch_size == targets_loader.batch_size, 'feature_loader and targets_loader must have the same batch size'

    # tracking best model
    best_loss = float('inf')
    best_model_state = None


    # logging
    losses = np.zeros(epochs)


    ffn.train()
    for epoch in tqdm(range(epochs), desc='Training progress', ):
        batch_losses = 0

        # mini-batching
        for batch_features, batch_targets in zip(feature_loader, targets_loader):
            batch_features = batch_features[0]
            batch_targets = batch_targets[0]

            optimizer.zero_grad()

            batch_predictions = ffn(batch_features)
            batch_loss = criterion(batch_predictions, batch_targets)

            # add cumulative loss for each batch (later averaged)
            batch_losses += batch_loss.item()


            batch_loss.backward()     # Backward pass: Compute gradient of the loss with respect to model parameters
            optimizer.step()    # Update weights
            scheduler.step() if scheduler else None    # Update learning rate

        # calculate mean loss for each batch
        loss = batch_losses / len(feature_loader)


        # loss print
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}') if epoch % (epochs // 10) == 0 else None

        # logging
        # if epoch % (epochs // 100) == 0 and epoch != 0:
        losses[epoch] = loss

        # save model at 1/10 of epochs
        if epoch % (epochs // 10) == 0:
            checkpoint_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch}.pt')
            torch.save(ffn.state_dict(), checkpoint_path)

        current_loss = loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = ffn.state_dict()


    # save best model
    best_model_path = os.path.join(exp_folder, 'best_model', 'best_model.pt')
    torch.save(best_model_state, best_model_path)

    # Save the logs as CSV
    log_df = pd.DataFrame({
        'Total Loss': losses,
    })
    log_csv_path = os.path.join(exp_folder, 'logs', 'training_log.csv')
    log_df.to_csv(log_csv_path, index=False)



#%% RUN
if __name__ == '__main__':

    # writer = SummaryWriter() # add on again when doing TensorBoard
    #! TODO: add JSON parsing and saving of hyperparameters in experiment folder

    parser = argparse.ArgumentParser(description='Run wave equation FFN experiment.')
    parser.add_argument('--exp_id', type=str, help='Optional experiment identifier (3-digit number).', default=None)
    mini_batching = True # implement JSON argument for this
    args = parser.parse_args()

    # handle missing exp_id argument
    exp_id = args.exp_id if args.exp_id else next_experiment_number('experiments')

    # create folders
    os.makedirs('experiments', exist_ok=True) # create experiments folder if it doesn't exist
    exp_folder = os.path.join('experiments', exp_id)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'best_model'), exist_ok=True)
    print(f'Experiment ID: {exp_id}')

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATA GENERATION
    # Static parameters
    boundaries = (-1,1)     # boundary values of x
    # T = 2                   # Time duration
    # num_samples = 1000      # Number of samples
    v = 1                   # wave speed

    ### Finite Difference Ground Truth Solution
    x_domain = (-1, 1)
    t_domain = (0, 2)  # Define T domain

    ground_truth,x_space,t_space = calc_ground_truth(x_domain, t_domain)


    batch_size = 64

    # data generation
    # x_samples = np.random.uniform(*boundaries, num_samples)
    # t_samples = np.random.uniform(0, T, num_samples)

    # Generate predictions over time and space
    X_mesh, T_mesh = torch.meshgrid(x_space, t_space, indexing='ij')
    feature_grid = torch.stack((X_mesh.flatten(), T_mesh.flatten()), dim=1)

    # features = np.column_stack((x_space, t_space)) # concatenate x and t

    features_ = torch.tensor(feature_grid, dtype=torch.float32, requires_grad=True, device=device)#.reshape(-1, 2)
    print(f'features shape: {features_.shape}')

    targets_ = torch.tensor(ground_truth, dtype=torch.float32, requires_grad=True, device=device).reshape(-1, 1)
    print(f'targets shape: {targets_.shape}')


    feature_dataset = TensorDataset(features_)
    targets_dataset = TensorDataset(targets_)
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=False)
    targets_loader = DataLoader(targets_dataset, batch_size=batch_size, shuffle=False)



    # defining model
    ffn = Net(input_dim=2, output_dim=1, hidden_dim=32, n_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ffn.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = optim.SGD(ffn.parameters(), lr=1e-3, momentum=0.9) 

    enable_scheduler = False
    # scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=0.5, verbose=True)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.001, total_iters=500, last_epoch=-1, verbose=False) if enable_scheduler else None

    ### Static parameters
    epochs = 5000


    train(ffn=ffn,
          optimizer=optimizer,
          criterion=criterion,
          feature_loader=feature_loader,
          targets_loader=targets_loader,
          scheduler=scheduler,
          boundaries=boundaries,
          v=v,
          epochs=epochs,
          exp_folder=exp_folder,
          device=device)

