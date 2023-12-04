import numpy as np
import pandas as pd
import os
import argparse
import re

import torch 
from torch import nn, optim
from torch.nn import functional as F

from softadapt import LossWeightedSoftAdapt


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


def initial_condition_1(model: nn.Module, features, device):
    # Gaussian source
    
    x0 = 0. # positions of the source
    sigma0 = 0.2 # width of the frequency 
    a = 1. # amplitude of the source

    x = features[:, 0] # take x-features from sampled data
    # find total number of samples in features
    t = torch.zeros_like(x, requires_grad=True, device=device) # set t=0 for all x
    coll_points = torch.column_stack((x, t)) # concatenate x and t

    pressure0 = a*torch.exp(-((x - x0)/sigma0)**2).reshape(-1,1).to(device)
    pred_pressure0 = model(coll_points)
    loss = torch.mean((pressure0 - pred_pressure0)**2)

    ##adder en punkt på toppen af gauss
    #collpoint_1 = torch.tensor([x0,0.]).reshape(1,2).to(device)
    #loss2 = (torch.tensor(1.).to(device)-model(collpoint_1))**2

    #loss3 = loss + loss2

    # print(f'IC1 loss: {loss.item()},\nPressure_true {pressure0[:10]},\nPressure_pred {pred_pressure0[:10]}') if print_values else None
    return loss, pressure0, pred_pressure0


def initial_condition_2(model: nn.Module, features, device):
    # v = 0 for t = 0 

    x = features[:, 0]
    t = torch.zeros_like(x, requires_grad=True).to(device)
    coll_points = torch.column_stack((x, t))

    pred_pressure0 = model(coll_points)
    pred_pressure0_grad, _ = gradient(pred_pressure0, t)
    loss = torch.mean(pred_pressure0_grad**2)

    # print(f'IC2 loss: {loss.item()}')
    return loss


def boundary_condition(model: nn.Module, features, device, boundaries: tuple):
    # boundary domain is in 'x'
    neg_bc, pos_bc = boundaries
    t = features[:, 1]
    x_neg_bc = torch.ones(len(t), dtype=torch.float32, requires_grad=True, device=device)*neg_bc
    x_pos_bc = torch.ones(len(t), dtype=torch.float32, requires_grad=True, device=device)*pos_bc

    # concat the boundary values to the time values
    neg_boundary = torch.column_stack((x_neg_bc, t))
    pos_boundary = torch.column_stack((x_pos_bc, t))

    neg_pred = model(neg_boundary)
    pos_pred = model(pos_boundary)
    
    neg_pred_grad, _ = gradient(neg_pred, neg_boundary)
    pos_pred_grad, _ = gradient(pos_pred, pos_boundary)
    
    loss = torch.mean(neg_pred_grad**2 + pos_pred_grad**2)

    # print(loss)
    return loss


def pde_loss(prediction: torch.tensor, features, device, v, epoch):
    """
    prediction: N length tensor of u(x, t)
    features: N length tensor of (x, t)
    v: wave speed (can also be called 'c')
    
    equation:
    u_tt - v**2 * u_xx = 0
    """
    u_grad_grad, _ = gradient(prediction, features, order=2)
    u_xx = u_grad_grad[:, 0]
    u_tt = u_grad_grad[:, 1]

    #compute the loss
    model_pde = (u_tt - v**2 * u_xx)


    # Create a tensor by broadcasting
    tensor = torch.ones(num_samples, num_samples).to(device)

    cutoff = int(num_samples * epoch / epochs)

    tensor[cutoff:, :] = 0.0

    model_pde = model_pde[:,] * tensor

    """
    # Create a linearly decreasing pattern from 1 to 0
    linear_pattern = 1.0 - torch.arange(1000, dtype=torch.float) / 999.0

    # Create a tensor by broadcasting the linear pattern across all columns
    tensor = torch.zeros(1000, 1000) + linear_pattern.view(-1, 1)

    model_pde = model_pde * tensor.to(device)
    """

    loss = torch.mean((model_pde)**2)
    # print(loss)
    return loss


def train(pinn, criterion, optimizer, features, boundaries, v, epochs, save_path, adaptive, device):
    """
    
    """

    # tracking best model
    best_loss = float('inf')
    best_model_state = None

    ### SoftAdapt parameters
    # Initializing adaptive weights to all ones.
    adapt_weights = torch.tensor([1,1,1,1]).to(device)
    # Change 1: Create a SoftAdapt object (with your desired variant)
    softadapt_object = LossWeightedSoftAdapt(beta=0.1)
    # Change 2: Define how often SoftAdapt calculate weights for the loss components
    epochs_to_make_updates = 5
    # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
    values_comp1 = []
    values_comp2 = []
    values_comp3 = []
    values_comp4 = []


    # logging
    losses = np.zeros(epochs)
    losses_pde = np.zeros(epochs)
    losses_ic1 = np.zeros(epochs)
    losses_ic2 = np.zeros(epochs)
    losses_bc = np.zeros(epochs)

    lambda_pde = np.zeros(epochs)
    lambda_ic1 = np.zeros(epochs)
    lambda_ic2 = np.zeros(epochs)
    lambda_bc = np.zeros(epochs)


    pinn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        if epoch == 8000:
            optimizer = optim.Adam(pinn.parameters(), lr=1e-5, weight_decay=1e-5)
        
        u_pred = pinn(features)

        # PDE physics loss
        loss_pde = pde_loss(u_pred, features, device, v=v, epoch=epoch)

        # print_IC1 = True if epoch % 499 == 0 and epoch != 0 else False
        # IC physics loss
        loss_ic1, pres0, pred_pres0 = initial_condition_1(pinn, features, device)
        loss_ic2 = initial_condition_2(pinn, features, device)

        # BC physics loss
        loss_bc = boundary_condition(pinn, features, device, boundaries=boundaries)

        
        values_comp1.append(loss_pde)
        values_comp2.append(loss_ic1)
        values_comp3.append(loss_ic2)
        values_comp4.append(loss_bc)
        
        # Change 4: Make sure `epochs_to_make_change` have passed before calling SoftAdapt.
        if epoch % epochs_to_make_updates == 0 and epoch != 0 and adaptive == True:
            adapt_weights = softadapt_object.get_component_weights(torch.tensor(values_comp1), 
                                                                    torch.tensor(values_comp2),
                                                                    torch.tensor(values_comp3),
                                                                    torch.tensor(values_comp4),
                                                                    verbose=False,
                                                                    )
                                    
            # Resetting the lists to start fresh (this part is optional)
            values_comp1 = []
            values_comp2 = []
            values_comp3 = []
            values_comp4 = []
        
        # Change 5: Update the loss function with the linear combination of all components.
        loss = adapt_weights[0]*loss_pde + adapt_weights[1]*loss_ic1 + adapt_weights[2]*loss_ic2 + adapt_weights[3]*loss_bc
        #loss = loss_ic1

        loss.backward()         # Backward pass: Compute gradient of the loss with respect to model parameters
        optimizer.step()        # Update weights
        
        
        # loss print
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}') if epoch % 100 == 0 and epoch != 0 else None
        print(f'loss components: \nloss_pde: {loss_pde.item()}, \nloss_ic1: {loss_ic1.item()}, \nloss_ic2: {loss_ic2.item()}, \nloss_bc: {loss_bc.item()}') if epoch % 100 == 0 and epoch != 0 else None
        # print(f'pressure 0 {pres0.mean()}\npred pressure 0 {pred_pres0.mean()}') if epoch % 100 == 0 and epoch != 0 else None

        # logging
        losses[epoch] = loss.item()
        losses_pde[epoch] = loss_pde.item()
        losses_ic1[epoch] = loss_ic1.item()
        losses_ic2[epoch] = loss_ic2.item()
        losses_bc[epoch] = loss_bc.item()

        lambda_pde[epoch] = adapt_weights[0]
        lambda_ic1[epoch] = adapt_weights[1]
        lambda_ic2[epoch] = adapt_weights[2]
        lambda_bc[epoch] = adapt_weights[3]

        # save model at 1/10 of epochs
        if epoch % (epochs // 10) == 0:
            checkpoint_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch}.pt')
            torch.save(pinn.state_dict(), checkpoint_path)

        current_loss = loss.item()
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
    })
    log_csv_path = os.path.join(exp_folder, 'logs', 'training_log.csv')
    log_df.to_csv(log_csv_path, index=False)





#%% RUN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run wave equation PINN experiment.')
    parser.add_argument('--exp_id', type=str, help='Optional experiment identifier (3-digit number).', default=None)
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
    T = 2                   # Time duration
    num_samples = 1000      # Number of samples
    v = 1                   # wave speed

    # data generation
    x_samples = np.random.uniform(*boundaries, num_samples)
    t_samples = np.random.uniform(0, T, num_samples)

    x_samples = np.sort(x_samples)
    t_samples = np.sort(t_samples)

    features = np.column_stack((x_samples, t_samples))
    features_ = torch.tensor(features, dtype=torch.float32, requires_grad=True).reshape(-1, 2).to(device)
    

    # defining model
    pinn = Net(input_dim=2, output_dim=1, hidden_dim=32, n_layers=2).to(device)
    criterion = nn.MSELoss() # actually not used but lets just keep it for now #TODO remove / or just use.
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer = optim.SGD(pinn.parameters(), lr=1e-3)

    ### Static parameters
    epochs = 20000

    print(f'Training with parameters \n{pinn}\n')
    train(pinn=pinn, 
          criterion=criterion, optimizer=optimizer, 
          features=features_, 
          boundaries=boundaries, v=v, 
          epochs=epochs, save_path=exp_folder,
          adaptive=True, device=device)

