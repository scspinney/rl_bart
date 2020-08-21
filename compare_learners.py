import numpy as np
import pandas as pd
from plots import plot_weights_bar
from utils import *
from maxent import get_stats
from plots import plot_reward_landscape

# get the feature matrices and trajectories from the human experts
maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
year=2

feature_matrices, Tprob, trajectories = load_data(maindir,year)


# get weights
human_weights = process_weights('results/theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','humans')
ai_weights1 = process_weights('results/QL_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','qlearner')
ai_weights2 = process_weights('results/AlwaysPump_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','alwayspump')
ai_weights3 = process_weights('results/Random_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','random')
ai_weights = ai_weights1.append(ai_weights2.append(ai_weights3))
all_weights = human_weights.append(ai_weights)

# bar plots of the weights
plot_weights_bar(all_weights)

# reward landscape plots
N_EXPERTS = 138
N_TRIAL = 30
N_STATES = 128
N_FEAT = 11

obs_exp_rewards, avg_save_state = get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories)

for player_type in all_weights.Player.unique():
    data = all_weights.query(f"Player == {player_type}")
    data=data.sort_values(by="Features")
    weights = data['Weights'].values
    plot_reward_landscape(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,weights,feature_matrices,obs_exp_rewards,avg_save_state)
