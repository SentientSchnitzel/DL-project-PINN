#%%
import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import torch

from train import Net
from utils import *



def extract_min_losses_and_hparams(exp_dir):
    runs = os.listdir(exp_dir)
    runs = [run for run in runs if os.path.isdir(os.path.join(exp_dir, run))] # filter out files
    
    losses_and_hparams = []
    for run in runs:
        path = os.path.join(exp_dir, run)
        config_path = os.path.join(path, "run_config.yml")
        log_path = os.path.join(path, "logs", "training_log.csv")
        
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

        log = pd.read_csv(log_path)
        log = log[log["Total Loss"] > 0.0] # filter out 0.0-losses from non-reached epochs
        min_loss = log["Total Loss"].min()

        # the config is two layered. Extract the second layer and create a flat dictionary
        hparams = {}
        for category in config:
            c = config[category]
            for key in c:
                hparams[key] = c[key]
        losses_and_hparams.append((min_loss, hparams))
    return losses_and_hparams

def get_relevant_hparams(config):
    relevant_hparams = ['learning_rate', 'hidden_dim', 'n_layers', 'dropout_rate', 'scheduler', 'adaptive', 'batch_size']
    return {key: config[key] for key in config if key in relevant_hparams}


def get_best_model(model, run_path, device):
     
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
    return model


#%%
if "__main__" == __name__:
    device = get_device()
    


    """
    ?find out which model is best / find top 10 models
        retrieve experiment folders
        loss_and_hp = []
        for each exp in experiments:,
            load configs_losses
            sort to 10 best models with hparams
            exp_loss_and_hp = [(loss, hparams),...]
            loss_and_hp.append(exp_loss_and_hp)
        concat all exp_lists
        
        sort to 10 best models with hparams

        plot.

    ?find out how hyperparameters affect the model's performance
    """
    # exps = ['127', '128', ]
    exps = ['132', '133', ] 
    exps_dirs = [os.path.join('experiments', exp) for exp in exps]
    
    # for exp_dir in exps_dirs:
    #     configs_losses = pd.read_csv(os.path.join(exp_dir, 'configs_losses.csv'), index_col=False)
    #     configs_losses = configs_losses[configs_losses["loss"] > 0.0] # filter out 0.0-losses from non-reached epochs #! change "loss" to "Total Loss" for new logs
    #     min_idx = configs_losses["loss"].argmin()
    #     print(f'Best run for {exp_dir}: \n{configs_losses}')


#%%
    
    # get the best model for each run. calc loss. 
    for exp_dir in exps_dirs:
        
        # get all runs
        runs = os.listdir(exp_dir)
        runs = [run for run in runs if os.path.isdir(os.path.join(exp_dir, run))] # filter out files
        for run in runs:
            run_path = os.path.join(exp_dir, run)
            run_conf_path = os.path.join(run_path, "run_config.yml")
            with open(run_conf_path, 'r') as stream:
                run_conf = yaml.safe_load(stream)
            
            # Define your model architecture (must match the saved model)
            model = Net(input_dim=2, 
                        output_dim=1, 
                        hidden_dim=run_conf['training']['hidden_dim'], 
                        n_layers=run_conf['training']['n_layers'],
                        dropout_rate=run_conf['training']['dropout_rate'],
                        initialization=run_conf['training']['initialization']
                        ).to(device)
            best_model = get_best_model(model, run_path, device)
            


#%%
    losses_and_hparams = {}
    for exp_dir in exps_dirs:
        exp_losses_and_hparams = extract_min_losses_and_hparams(exp_dir)
        # only save the hparams that are tuned
        exp_losses_and_hparams = [(loss, get_relevant_hparams(hparams)) for loss, hparams in exp_losses_and_hparams]
        # find the runs with the 10 lowest losses, their index and their config
        losses = [loss for loss, _ in exp_losses_and_hparams] # create the losses list we can sort
        min_indices = np.argsort(losses)[:10] # get the indices of the 10 lowest losses
        top10 = [exp_losses_and_hparams[i] for i in min_indices] # extract the 10 lowest losses and their params
        
        losses_and_hparams[exp_dir] = top10
        

    # find the ultimate top 10
    all_losses_hparams = []
    for exp_dir, loss_hparam_pairs in losses_and_hparams.items():
        all_losses_hparams.extend(loss_hparam_pairs) # unpacks loss-hparam pairs into a list
    top_10 = sorted(all_losses_hparams, key=lambda x: x[0])[:10]
    top_10_losses = [pair[0] for pair in top_10]
    top_10_hparams = [pair[1] for pair in top_10]

    # Plot the top 10 losses
    plt.figure(figsize=(16, 8), dpi=100)
    plt.barh(range(len(top_10_losses)), top_10_losses)
    plt.xlabel("Minimum Loss")
    plt.xscale("log")
    plt.ylabel("Run Index")
    # Annotate the bar plot with loss values
    for i, v in enumerate(top_10_losses):
        plt.text(v, i - 0.1, f"{v:.2e}")
    plt.title(f"10 best run configs by loss. BS: {1}, Adaptive: false")
    plt.show()

    [print(f'Best 10 runs: loss: {loss} with {hp}') for loss, hp in top_10]