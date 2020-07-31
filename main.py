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
N=148
n_epochs=1
n_iters=400
lr=1E-3
lr_decay=1
gamma=0.99

#popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

feature_matrices, Tprob, trajectories = load_data(maindir,year,N)

N_EXPERTS = len(feature_matrices)
N_TRIAL, N_STATES, N_FEAT = np.shape(feature_matrices[0])
N_STATES -= 2

# fit IRL
maxent_irl(maindir, year, feature_matrices, Tprob, gamma, trajectories, lr,lr_decay,n_iters,n_epochs,shuffle_training=True,use_prior=False)

print("Successfully ran MaxEnt IRL algorithm.")







