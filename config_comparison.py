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
    exps = ['127', '128', ]
    exps_dirs = [os.path.join('experiments', exp) for exp in exps]
    
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
        

    # find the ultimate top 10 by sorting the losses_and_hparams dict by the value
    
    

    # horizontal bar chart of the 10 runs with the lowest losses
    plt.figure(figsize=(16,8), dpi=100)
    plt.barh(range(len(min_losses)), min_losses)
    plt.xlabel("Minimum Loss")
    plt.xscale("log")
    plt.ylabel("Run")
    # insert the minimum loss values as text next to the bars
    for i, v in enumerate(min_losses):
        plt.text(v, i - 0.1, f"{v:.2e}")
    # plt.xlim(0, max(min_losses) * 1.1)
    # insert name of the run as y-tick labels
    plt.yticks(range(len(min_runs)), min_runs)


    plt.title(f"10 best run configs by loss. BS: {1}, Adaptive: false")
    plt.show()
        
        # min_losses = [losses[i] for i in min_indices] # extract the 10 lowest losses
        # min_runs = [runs[i] for i in min_indices] # extract the dir names --||--
        # min_configs = [os.path.join(exp_dir, run, "run_config.yml") for run in min_runs] # extract the config paths