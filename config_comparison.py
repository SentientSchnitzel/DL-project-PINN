import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import torch

from utils import *
from train import Net



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





if "__main__" == __name__:

    


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
    exps = ['132'] # ['127', '128', ]
    exps_dirs = [os.path.join('experiments', exp) for exp in exps]
    
    for exp_dir in exps_dirs:
        configs_losses = pd.read_csv(os.path.join(exp_dir, 'configs_losses.csv'), index_col=False)
        configs_losses = configs_losses[configs_losses["loss"] > 0.0] # filter out 0.0-losses from non-reached epochs
        min_idx = configs_losses["loss"].argmin()
        print(f'Best run for {exp_dir}: \n{configs_losses}')


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

    print(f'Best 10 runs: {[pair for pair in top_10_hparams]}')