"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import math
import numpy as np
from itertools import product


def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.
    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def optimal_policy(Tprob, rewards, gamma, error=0.01,stochastic=True):
    """
    static value iteration function. Perhaps the most useful function in this repo

    inputs:
      Tprob         NxAxN_ACTIONS transition probabilities matrix -
                                P_a[s0, a, s1] is the transition prob of
                                landing at state s1 when taking action
                                a at state s0
      rewards     Nx1 matrix - rewards for all the states
      gamma       float - RL discount
      error       float - threshold for a stop

    returns:
      values    Nx1 matrix - estimated values
      policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
    """
    N_STATES, N_ACTIONS, _ = np.shape(Tprob)

    values = np.zeros(N_STATES)

    # estimate values
    while True:
        values_tmp = values.copy()

        for s in range(N_STATES):
            v_s = []
            values[s] = max(
                [sum([Tprob[s, a, s1] * (rewards[s] + gamma * values_tmp[s1]) for s1 in range(N_STATES)]) for a in
                 range(N_ACTIONS)])

        if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
            break

    policy = np.zeros([N_STATES, N_ACTIONS])
    for s in range(N_STATES):
        v_s = np.array([sum([Tprob[s, a, s1] * (rewards[s] + gamma * values[s1]) for s1 in range(N_STATES)]) for a in
                        range(N_ACTIONS)])
        policy[s,:] = np.transpose(v_s / np.sum(v_s))

    # terminal, no choices
    policy[-2:,:] = 0

    return values, policy



def compute_state_visition_freq(Tprob, gamma, trajectory, policy):
    """compute the expected states visition frequency p(s| theta, T)
    using dynamic programming
    inputs:
      Tprob     NxAxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES, N_ACTIONS, _  = np.shape(Tprob)

    #trajs = np.reshape(trajs,-1)
    trajectory_length = len(trajectory)

    start_state_count = np.zeros(N_STATES)
    start_state_count[0] = 1

    expected_svf = start_state_count

    state=0
    while True:
        pumpp = policy[state,0] # prob of pumping
        pump = pump_prob(pumpp)

        if pump: # chose to pump
            pop = np.random.binomial(1,Tprob[state,0,-1],1)
            if pop:
                expected_svf[-1]+=1
                break
            else:
                expected_svf[state+1]+=1
                state+=1
        else: # chose to save
            expected_svf[-2]+=1
            break

    return expected_svf
    # expected_svf = np.tile(p_start_state, (30, 1)).T
    #
    # for t in range(1, 30):
    #     expected_svf[:, t] = 0
    #     for i in range(N_STATES):
    #         for j in range(N_ACTIONS):
    #             for k in range(N_STATES):
    #     #for i, j, k in product(range(N_STATES), range(N_ACTIONS), range(N_STATES)):
    #                 tmp = expected_svf[k, t - 1] *policy[i, j] * Tprob[i, j, k]
    #                 expected_svf[i, t] += tmp


    #return expected_svf.sum(axis=1)



def pump_prob(p):

    if p == 0:
        return 0.0001
    elif p == 1:
        return 0.9999
    else:
        return np.random.binomial(1,p,1)