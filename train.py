import numpy as np
import pandas as pd
import os
import argparse
import re
import yaml
import itertools


import torch 
from torch import nn, optim
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, OneCycleLR, CosineAnnealingWarmRestarts, CyclicLR

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
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout_rate, initialization):
        
        super(Net, self).__init__()

        activation = nn.Softplus # nn.LeakyReLU, nn.Tanh, nn.ReLU, nn.ELU

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            activation(),
            nn.Dropout(dropout_rate)
        )
        
        self.network = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout_rate)
            ) for i in range(n_layers)]
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
        # He/Kaiming init for relu descendent softplus
        if initialization:
            init.kaiming_normal_(self.input[0].weight, nonlinearity='relu')
            for layer in self.network:
                init.kaiming_normal_(layer[0].weight, nonlinearity='relu')
            init.kaiming_normal_(self.output[0].weight, nonlinearity='linear')

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


def initial_condition_1(model: nn.Module, batch_features, x0, sigma, amplitude, device):

    x = batch_features[:, 0] # take x-features from batch
    t = torch.zeros_like(x, requires_grad=True, device=device) # set t=0 for all x
    coll_points = torch.column_stack((x, t)) # concatenate x and t

    pressure0 = amplitude*torch.exp(-((x - x0)/sigma)**2).to(device).reshape(-1, 1)
    pred_pressure0 = model(coll_points)
    loss = torch.mean((pressure0 - pred_pressure0)**2)

    # print(f'IC1 loss: {loss.item()},\nPressure_true {pressure0[:10]},\nPressure_pred {pred_pressure0[:10]}') if print_values else None
    return loss #pressure0, pred_pressure0


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
    # random subsample of t
    t = t[torch.randperm(len(t))[:len(t)//2]]

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


def generate_dataloader(x_domain, t_domain, conf, device):
    """
    Generates data for the wave equation PINN.
    """

    num_samples = conf['data']['num_collocation_points']
    batch_size = conf['data']['batch_size'] if conf['data']['batch_size'] != -1 else num_samples # -1 batch size mean no minibatching

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


def train(pinn, optimizer, dataloader, boundaries, epochs, exp_folder, adaptive, device, scheduler, n_logs, physics, patience):
    """
    Trains a Physics Informed Neural Network (PINN).

    Args:
    pinn: the PINN model
    optimizer: optimizer object
    dataloader: torch DataLoader object containing batches
    boundaries: the boundary conditions for the PDE problem.
    epochs: the number of epochs to train for.
    exp_folder: experiment folder path
    adaptive: whether to use adaptive weights or not
    device: torch device
    scheduler: learning rate scheduler object
    n_logs: number of loggings throughout training
    physics: variables for the physics of the problem (wave speed, sigma, x0, amplitude)
    patience: early stopping patience
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


    # unpack physics
    v = physics['wave_speed']
    sigma = physics['sigma']
    x0 = physics['x0']
    amplitude = physics['amplitude']

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
            batch_loss_ic1 = initial_condition_1(pinn, batch_features, x0, sigma, amplitude, device)
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

        # save model at 1/5 of epochs
        if epoch % (epochs // 5) == 0 and epoch != 0:
            checkpoint_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch}.pt')
            torch.save(pinn.state_dict(), checkpoint_path)

        # save best model and early stopping criteria
        current_loss = loss_total
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = pinn.state_dict()
            
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}.')
                break


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

    return best_loss


def setup_model(Net, conf):
    """
    Setup model from config dict.
    """
    pinn = Net(input_dim=2, 
                output_dim=1, 
                hidden_dim=conf['training']['hidden_dim'], 
                n_layers=conf['training']['n_layers'],
                dropout_rate=conf['training']['dropout_rate'],
                initialization=conf['training']['initialization'],
                ).to(device)
    optimizer = get_optimizer(pinn, conf)
    scheduler = get_scheduler(optimizer, conf)

    return pinn, optimizer, scheduler

    
def get_scheduler(optimizer, conf):
    """
    Get scheduler from a string-name.
    """

    epochs = conf['training']['epochs']
    if conf['data']['batch_size'] != -1: # if batch_size is not -1, we use some size of mini-batches
        samples = conf['data']['num_collocation_points']
        batch_size = conf['data']['batch_size']
        steps_per_epoch = int(np.ceil(samples/batch_size))
        one_cycle = int(steps_per_epoch * epochs/5)
    else:
        steps_per_epoch = 1
        one_cycle = int(epochs/5)


    scheduler_name = conf['training']['scheduler']
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=0.5, verbose=True)
    elif scheduler_name == 'LinearLR':
        scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=2500, last_epoch=-1, verbose=False)
    elif scheduler_name == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer=optimizer, max_lr=conf['training']['learning_rate'], 
                               epochs=epochs, steps_per_epoch=steps_per_epoch, 
                               pct_start=0.3, anneal_strategy='cos', last_epoch=-1, verbose=False)
    elif scheduler_name == 'CosineAnnealingWarmRestarts' or scheduler_name == 'CAWR': # accept both long and short name
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=one_cycle, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    elif scheduler_name == 'CyclicLR' or scheduler_name == 'Cycl': # accept both long and short name
        scheduler = CyclicLR(optimizer=optimizer, base_lr=conf['training']['learning_rate']/10, max_lr=conf['training']['learning_rate'], 
                             step_size_up=int(one_cycle/2), mode='triangular2', cycle_momentum=False, last_epoch=-1, verbose=False,)
    else:
        scheduler = None
    return scheduler


def get_optimizer(model: nn.Module, conf):
    """
    Get the optimizer from a string-name.
    """
    optimizer_name = conf['training']['optimizer']
    L2_penalty = conf['training']['L2_penalty']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=conf['training']['learning_rate'], weight_decay=L2_penalty)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=conf['training']['learning_rate'], momentum=0.9, weight_decay=L2_penalty)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not implemented. Please choose from: Adam, SGD.')
    return optimizer

def setup_tuneable_configurations(conf):
    """
    Setup tuning from config.
    """
    # adaptive and batch_size tunings are run as overall experiments, so we can distribute 
    # the work to different machines and get results faster.

    # define shorthand for config file names
    param_names = [
    'lr',       # learning rate
    'hdim',     # hidden dimension
    'lay',      # number of layers
    'rate',     # dropout rate 
    'sched',    # scheduler
    # 'adaptive', 
    # 'batch_size', 
    ]
    
    configurations = [
    dict(zip(param_names, config))
    for config in itertools.product(
        conf['training']['learning_rate'],
        conf['training']['hidden_dim'],
        conf['training']['n_layers'],
        conf['training']['dropout_rate'],
        conf['training']['scheduler'],
        # conf['training']['adaptive'],
        # conf['data']['batch_size'],
        )
    ]

    return configurations, param_names


def create_dir_name(config):
    # Create a list of formatted key-value pairs
    key_value_pairs = [f"{key}_{str(value).replace('.', '_')}" for key, value in config.items()]
    
    # Join the pairs into a single config-string separated by underscores
    config_name = "_".join(key_value_pairs)
    return config_name

def update_config_file(base_config, update_config, config_path):
    """
    updates the baseline configuration with the new hyperparameters for a specific run.
    """
    # overwrite the config yaml file for this specific run
    base_config['training']['learning_rate'] = update_config['lr']
    base_config['training']['hidden_dim'] = update_config['hdim']
    base_config['training']['n_layers'] = update_config['lay']
    base_config['training']['dropout_rate'] = update_config['rate']
    base_config['training']['scheduler'] = update_config['sched']
    # save the config file in the folder for this run.
    with open(os.path.join(config_path, 'run_config.yml'), 'w') as file:
        yaml.dump(base_config, file)
    return base_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run wave equation PINN experiment.')
    args = get_arguments(parser)
    base_conf = get_config(args.conf_path)

    # handle missing exp_id argument
    exp_id = args.exp_id if args.exp_id else next_experiment_number('experiments')

    # create folders
    exp_folder = create_exp_folder(exp_id) # returns the experiment folder path
    print(f"With batch size : {base_conf['data']['batch_size']} and adaptive : {base_conf['training']['adaptive']}")
    
    device = get_device(verbose=True)

    # unpack feature domains x and t
    x_domain = base_conf['data']['x_domain']
    t_domain = base_conf['data']['t_domain']
    run_conf = base_conf.copy() # initialize run_conf with the same values as conf, which later will be overwritten
    # save the config file in the folder for this experiment.
    with open(os.path.join(exp_folder, 'exp_config.yml'), 'w') as file:
        yaml.dump(base_conf, file)

    configurations, param_names = setup_tuneable_configurations(base_conf)
    configs_losses = {}
    ### Tuning loop
    for i, config in enumerate(configurations):
        # say which configuration we are running
        print(f'Configuration {i} with hyperparameters: {config}')
        
        # create appropriate folders for the config
        config_name = create_dir_name(config)
        config_path = os.path.join(exp_folder, config_name)
        os.makedirs(config_path, exist_ok=True) # make the directory
        create_config_dir_structure(config_path) # logs, checkpoints, figures, best_model
        run_conf = update_config_file(base_config=run_conf, update_config=config, config_path=config_path)
        dataloader = generate_dataloader(x_domain=x_domain,
                                        t_domain=t_domain,
                                        conf=run_conf,
                                        device=device)
        
        ### MODEL SPECIFICATION
        pinn, optimizer, scheduler = setup_model(Net, run_conf)

        # TensorBoard writer
        tensorboard = run_conf['logging']['tensorboard']
        writer = SummaryWriter(os.path.join('runs', exp_id)) if tensorboard else None

        print(f'Training with parameters, \n{pinn}\noptimizer, \n{optimizer}\nand learning rate scheduler, \n{scheduler} ')
        best_loss = train(pinn=pinn, 
                        optimizer=optimizer, 
                        dataloader=dataloader, 
                        boundaries=x_domain, 
                        epochs=run_conf['training']['epochs'], 
                        exp_folder=config_path,
                        adaptive=run_conf['training']['adaptive'], 
                        device=device,
                        scheduler=scheduler, 
                        n_logs=run_conf['logging']['n_logs'],
                        physics=run_conf['physics'],
                        patience=run_conf['training']['patience'],
                        )
        configs_losses[config_name] = best_loss.item()

        writer.close() if tensorboard else None
        print(f'Configuration {i} finished. Lowest loss: {best_loss} from configuration {config_path}')
    
    # save the results of the tuning
    configs_losses_df = pd.DataFrame.from_dict(configs_losses, orient='index', columns=['Total Loss'])
    configs_losses_df.to_csv(os.path.join(exp_folder, 'configs_losses.csv'))
    print(f'Experiment {exp_id} finished. Lowest loss: {best_loss} from configuration {config_path}')