import numpy as np
import pandas as pd
import os
import argparse
import re
import yaml

import torch 
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR, OneCycleLR

from tqdm import tqdm

from softadapt import LossWeightedSoftAdapt

from utils import *


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
                activation()
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


def gradient(outputs, inputs, order=1):
    """
    Computes the partial derivative of 
    an output with respect to an input.
    Given an order, we compute the gradient multiple times.
    """
    grads = []
    for i in range(order):
        grads.append(torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0])
        outputs = grads[-1]
    
    return outputs, grads


def initial_condition_1(model: nn.Module, batch_features, device):
    # Gaussian source
    
    x0 = 0 # positions of the source
    sigma0 = 0.2 # width of the frequency
    a = 1 # amplitude of the source

    x = batch_features[:, 0] # take x-features from batch
    t = torch.zeros_like(x, requires_grad=True, device=device) # set t=0 for all x
    coll_points = torch.column_stack((x, t)) # concatenate x and t

    pressure0 = a*torch.exp(-((x - x0)/sigma0)**2).to(device).reshape(-1, 1)
    pred_pressure0 = model(coll_points)
    loss = torch.mean((pressure0 - pred_pressure0)**2)

    # print(f'IC1 loss: {loss.item()},\nPressure_true {pressure0[:10]},\nPressure_pred {pred_pressure0[:10]}') if print_values else None
    return loss, pressure0, pred_pressure0


def initial_condition_2(model: nn.Module, batch_features, device):
    # v = 0 for t = 0 

    x = batch_features[:, 0]
    t = torch.zeros_like(x, requires_grad=True).to(device)
    coll_points = torch.column_stack((x, t))

    pred_pressure0 = model(coll_points)
    pred_pressure0_grad, _ = gradient(pred_pressure0, t)
    loss = torch.mean(pred_pressure0_grad**2)

    # print(f'IC2 loss: {loss.item()}')
    return loss


def boundary_condition(model: nn.Module, batch_features, device, boundaries: tuple):
    # boundary domain is in 'x'
    neg_bc, pos_bc = boundaries
    t = batch_features[:, 1]
    x_neg_bc = torch.ones(len(t), dtype=torch.float32, requires_grad=True, device=device)*neg_bc
    x_pos_bc = torch.ones(len(t), dtype=torch.float32, requires_grad=True, device=device)*pos_bc

    # concat the boundary values to the time values
    neg_boundary = torch.column_stack((x_neg_bc, t))
    pos_boundary = torch.column_stack((x_pos_bc, t))

    neg_pred = model(neg_boundary)
    pos_pred = model(pos_boundary)
    
    neg_pred_grad, _ = gradient(neg_pred, x_neg_bc)
    pos_pred_grad, _ = gradient(pos_pred, x_pos_bc)
    
    loss = torch.mean(neg_pred_grad**2 + pos_pred_grad**2)

    # print(loss)
    return loss


def pde_loss(model: nn.Module, batch_f, device, v, epoch, epochs):
    """
    prediction: N length tensor of u(x, t)
    batch_features: N length tensor of (x, t)
    v: wave speed (can also be called 'c')
    
    equation:
    u_tt - v**2 * u_xx = 0
    """

    # cutoff = int(batch_f.shape[0] * epoch / epochs)
    # batch_f = batch_f[:cutoff, :] #! TODO, nukes the data/values to NaN...
    prediction = model(batch_f)
    
    u_grad_grad, _ = gradient(prediction, batch_f, order=2)
    u_xx = u_grad_grad[:, 0]
    u_tt = u_grad_grad[:, 1]

    #compute the loss
    model_pde = (u_tt - v**2 * u_xx)
    loss = torch.mean((model_pde)**2)

    # print(loss)
    return loss

def generate_dataloader(x_domain, t_domain, num_samples, v, batch_size, device):
    """
    Generates data for the wave equation PINN.
    """
    # data generation
    x_samples = np.random.uniform(*x_domain, num_samples)
    t_samples = np.random.uniform(*t_domain, num_samples)

    features = np.column_stack((x_samples, t_samples)) # concatenate x and t
    features_ = torch.tensor(features, dtype=torch.float32, requires_grad=True, device=device).reshape(-1, 2)
    # sort features by second column (time) and keep pairings of observations.
    # features_ = features_[torch.argsort(features_[:, 1])].clone()
    dataset = TensorDataset(features_)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def train(pinn, criterion, optimizer, dataloader, boundaries, v, epochs, exp_folder, adaptive, device, scheduler, n_logs):
    """
    pinn:       PINN model
    criterion:  loss function
    optimizer:  optimizer
    dataloader: torch dataloader
    boundaries: tuple of boundary values
    v:          wave speed
    epochs:     number of epochs
    save_path:  path to save the model
    """

    # tracking best model
    best_loss = float('inf')
    best_model_state = None

    ### SoftAdapt parameters
    # Initializing adaptive weights to all ones.
    adapt_weights = torch.tensor([1,1,1,1]).to(device)
    # Change 1: Create a SoftAdapt object (with your desired variant)
    softadapt_object = LossWeightedSoftAdapt(beta=0.3)
    # Change 2: Define how often SoftAdapt calculate weights for the loss components
    epochs_to_make_updates = 5
    # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
    values_comp1 = []
    values_comp2 = []
    values_comp3 = []
    values_comp4 = []


    # logging
    n_logs = n_logs
    log_every_n_epochs = epochs // n_logs

    losses = np.zeros(n_logs)
    losses_pde = np.zeros(n_logs)
    losses_ic1 = np.zeros(n_logs)
    losses_ic2 = np.zeros(n_logs)
    losses_bc = np.zeros(n_logs)

    lambda_pde = np.zeros(n_logs)
    lambda_ic1 = np.zeros(n_logs)
    lambda_ic2 = np.zeros(n_logs)
    lambda_bc = np.zeros(n_logs)

    learning_rates = np.zeros(n_logs)


    pinn.train()
    for epoch in tqdm(range(epochs), desc='Training progress', ):
        log_index = epoch // log_every_n_epochs
        batch_PDE_losses = 0
        batch_IC1_losses = 0
        batch_IC2_losses = 0
        batch_BC_losses = 0

        # mini-batching
        for batch_features in dataloader:
            batch_features = batch_features[0]
        
            optimizer.zero_grad()
            
            # u_pred = pinn(batch_features)

            # PDE physics loss
            batch_loss_pde = pde_loss(pinn, batch_features, device, v=v, epoch=epoch, epochs=epochs)

            # IC physics loss
            batch_loss_ic1, pres0, pred_pres0 = initial_condition_1(pinn, batch_features, device)
            # TODO: find a way to concat all pred_pres0 to plot their behaviour later
            batch_loss_ic2 = initial_condition_2(pinn, batch_features, device)

            # BC physics loss
            batch_loss_bc = boundary_condition(pinn, batch_features, device, boundaries=boundaries)

            # add cumulative loss for each batch (later averaged)
            batch_PDE_losses += batch_loss_pde.item()
            batch_IC1_losses += batch_loss_ic1.item()
            batch_IC2_losses += batch_loss_ic2.item()
            batch_BC_losses += batch_loss_bc.item()
            

            # Update the loss function with the SoftAdapt weighted components
            loss = adapt_weights[0]*batch_loss_pde + adapt_weights[1]*batch_loss_ic1 + adapt_weights[2]*batch_loss_ic2 + adapt_weights[3]*batch_loss_bc
            # loss = batch_loss_ic1

            loss.backward()     # Backward pass: Compute gradient of the loss with respect to model parameters
            optimizer.step()    # Update weights
            scheduler.step() if scheduler else None
        
        # calculate mean loss for each batch
        loss_pde = batch_PDE_losses / len(dataloader)
        loss_ic1 = batch_IC1_losses / len(dataloader)
        loss_ic2 = batch_IC2_losses / len(dataloader)
        loss_bc = batch_BC_losses / len(dataloader)
        
        loss_total = adapt_weights[0]*loss_pde + adapt_weights[1]*loss_ic1 + adapt_weights[2]*loss_ic2 + adapt_weights[3]*loss_bc

        values_comp1.append(loss_pde)
        values_comp2.append(loss_ic1)
        values_comp3.append(loss_ic2)
        values_comp4.append(loss_bc)
        
        # Make sure `epochs_to_make_change` have passed before calling SoftAdapt.
        if epoch % epochs_to_make_updates == 0 and epoch != 0 and adaptive == True:
            adapt_weights = softadapt_object.get_component_weights(torch.tensor(values_comp1), 
                                                                    torch.tensor(values_comp2),
                                                                    torch.tensor(values_comp3),
                                                                    torch.tensor(values_comp4),
                                                                    verbose=False,
                                                                    )         
            # Resetting the lists to start fresh
            values_comp1 = []
            values_comp2 = []
            values_comp3 = []
            values_comp4 = []
        

        # loss print
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_total}') if epoch % 100 == 0 and epoch != 0 else None
        print(f'loss components: \nloss_pde: {loss_pde}, \nloss_ic1: {loss_ic1}, \nloss_ic2: {loss_ic2}, \nloss_bc: {loss_bc}') if epoch % 100 == 0 and epoch != 0 else None
        # print(f'pressure 0 {pres0.mean()}\npred pressure 0 {pred_pres0.mean()}') if epoch % 100 == 0 and epoch != 0 else None

        # logging
        if epoch % log_every_n_epochs == 0:
            losses[log_index] = loss_total
            losses_pde[log_index] = loss_pde
            losses_ic1[log_index] = loss_ic1
            losses_ic2[log_index] = loss_ic2
            losses_bc[log_index] = loss_bc

            lambda_pde[log_index] = adapt_weights[0]
            lambda_ic1[log_index] = adapt_weights[1]
            lambda_ic2[log_index] = adapt_weights[2]
            lambda_bc[log_index] = adapt_weights[3]

            learning_rates[log_index] = optimizer.param_groups[0]['lr']

        if tensorboard:
            # Log scalar values to the tensorboard writer.
            writer.add_scalar('Loss/Total', loss_total, epoch)
            writer.add_scalar('Loss/PDE', loss_pde, epoch)
            writer.add_scalar('Loss/IC1', loss_ic1, epoch)
            writer.add_scalar('Loss/IC2', loss_ic2, epoch)
            writer.add_scalar('Loss/BC', loss_bc, epoch)

            writer.add_scalar('Lambda/PDE', adapt_weights[0], epoch)
            writer.add_scalar('Lambda/IC1', adapt_weights[1], epoch)
            writer.add_scalar('Lambda/IC2', adapt_weights[2], epoch)
            writer.add_scalar('Lambda/BC', adapt_weights[3], epoch)

        # save model at 1/10 of epochs
        if epoch % (epochs // 10) == 0 and epoch != 0:
            checkpoint_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch}.pt')
            torch.save(pinn.state_dict(), checkpoint_path)

        current_loss = loss_total
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = pinn.state_dict()


    # save best model
    best_model_path = os.path.join(exp_folder, 'best_model', 'best_model.pt')
    torch.save(best_model_state, best_model_path)

    # Save the logs as CSV
    log_df = pd.DataFrame({
        'Total Loss': losses,
        'PDE Loss': losses_pde,
        'IC1 Loss': losses_ic1,
        'IC2 Loss': losses_ic2,
        'BC Loss': losses_bc,
        'lambda PDE': lambda_pde,
        'lambda IC1': lambda_ic1,
        'lambda IC2': lambda_ic2,
        'lambda BC': lambda_bc,
        'Learning Rate': learning_rates,
    })
    log_csv_path = os.path.join(exp_folder, 'logs', 'training_log.csv')
    log_df.to_csv(log_csv_path, index=False)


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

def setup_model(conf): #! TODO
    """
    Setup model from config.
    """

    # scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=0.5, verbose=True)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=2500, last_epoch=-1, verbose=False)

    pass

def create_folder_structure(exp_id): #! TODO
    """
    Create folder structure for experiment.
    """
    pass

#%% RUN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run wave equation PINN experiment.')
    args = get_arguments(parser)


    # handle missing exp_id argument
    exp_id = args.exp_id if args.exp_id else next_experiment_number('experiments')

    conf = get_config(args.conf_path)
    tensorboard = conf['logging']['tensorboard']
    n_logs = conf['logging']['n_logs']


    # create folders
    os.makedirs('experiments', exist_ok=True) # create experiments folder if it doesn't exist
    exp_folder = os.path.join('experiments', exp_id)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'best_model'), exist_ok=True)
    print(f'Experiment ID: {exp_id}')

    # set device
    device = get_device(verbose=True)

    
    x_domain = conf['data']['x_domain']
    t_domain = conf['data']['t_domain']
    v = conf['data']['wave_speed']
    num_samples = conf['data']['num_collocation_points']
    batch_size = conf['data']['batch_size'] if conf['data']['batch_size'] != -1 else num_samples # -1 batch size mean no minibatching

    # data generation
    dataloader = generate_dataloader(x_domain=x_domain,
                                     t_domain=t_domain,
                                     num_samples=num_samples, 
                                     v=v, 
                                     batch_size=batch_size, 
                                     device=device)


    ### MODEL HYPERPARAMETERS
    epochs = conf['training']['num_epochs']

    pinn = Net(input_dim=2, output_dim=1, hidden_dim=32, n_layers=2).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=1e-2)

    scheduler = None
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=1e-2, epochs=2000, steps_per_epoch=int(np.floor(num_samples/batch_size)+1), pct_start=0.3, anneal_strategy='cos', last_epoch=-1, verbose=False)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join('runs', exp_id)) if tensorboard else None

    print(f'Training with parameters \n{pinn}\n')
    train(pinn=pinn, 
          criterion=None, optimizer=optimizer, 
          dataloader=dataloader, 
          boundaries=x_domain, v=v, 
          epochs=epochs, exp_folder=exp_folder,
          adaptive=True, device=device,
          scheduler=scheduler, n_logs=n_logs)

    writer.close() if tensorboard else None