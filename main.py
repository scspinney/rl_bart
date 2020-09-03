import numpy as np
import glob as glob
import os
from maxent import *
from likelihood import *
from plots import *
import matplotlib.pyplot as plt
from utils import *

maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
year=2
N=138
n_epochs=100
n_iters=1
lr=1E-4
lr_decay=1
gamma=1
seed=100

#popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

ai=True
feature_matrices, Tprob, trajectories = load_data(maindir,year)

if ai: #overwrite human expert with ai
    ai_type = 'Optimal' 
    feature_matrices = np.load(f'data/agents/feature_matrices{ai_type}N138T30S128F11.npy',allow_pickle=True)
    trajectories = np.load(f'data/agents/trajectories{ai_type}N138T30S128F11.npy',allow_pickle=True)

else:
    ai_type=""
# fit IRL
maxent_irl(maindir, N, year, feature_matrices, Tprob, gamma, trajectories, lr,lr_decay,n_iters,n_epochs,seed,ai,ai_type,shuffle_training=True,use_prior=False)

print("Successfully ran MaxEnt IRL algorithm.")







