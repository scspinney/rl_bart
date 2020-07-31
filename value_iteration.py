"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import math
import numpy as np
from itertools import product
from numba import jit,njit


def compute_state_visition_freq(N_STATES,Tprob,policy):
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

    start_state_count = np.zeros(N_STATES)
    start_state_count[0] = 1
    #expected_svf = start_state_count
    expected_svf = np.zeros(N_STATES)
    expected_svf[0] = 1

    while True:

        tmp_exp_svf = start_state_count.copy()
        for s in range(N_STATES-1):

            # pump no pop
            tmp_exp_svf[s+1] = tmp_exp_svf[s+1] + expected_svf[s]*policy[s,0]*Tprob[s,0,s+1]

            # pump and pop
           # tmp_exp_svf[-1] = tmp_exp_svf[-1] + expected_svf[s]*policy[s, 0]*Tprob[s,0,-1]

            # save
            #tmp_exp_svf[-2] = tmp_exp_svf[-2] + expected_svf[s]*policy[s, 1] * Tprob[s,1,-2]

        if all(abs(tmp_exp_svf - expected_svf) < 0.1):
            break
        else:
            expected_svf = tmp_exp_svf

    return expected_svf


@njit
def compute_state_visition_freq_jit(N_STATES,Tprob,policy):
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

    start_state_count = np.zeros(N_STATES)
    start_state_count[0] = 1
    expected_svf = np.zeros(N_STATES)
    expected_svf[0] = 1

    while True:

        tmp_exp_svf = start_state_count.copy()
        for s in range(N_STATES-1):

            # pump no pop
            tmp_exp_svf[s+1] = tmp_exp_svf[s+1] + expected_svf[s]*policy[s,0]*Tprob[s,0,s+1]

            # pump and pop
           # tmp_exp_svf[-1] = tmp_exp_svf[-1] + expected_svf[s]*policy[s, 0]*Tprob[s,0,-1]

            # save
            #tmp_exp_svf[-2] = tmp_exp_svf[-2] + expected_svf[s]*policy[s, 1] * Tprob[s,1,-2]

        diffs = tmp_exp_svf - expected_svf
        notfinished=0
        for i in range(diffs.shape[0]):
            if abs(diffs[i]) < 0.1:
                continue
            else:
                notfinished = 1
                break

        if notfinished:
            expected_svf = tmp_exp_svf
        else:
            break

    return expected_svf



@njit
def optimal_value_jit(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-3):
    """
    Find the optimal value function.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = -1000000
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = -1000000
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v



def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v



def find_policy(n_states, r, n_actions, discount,
                           transition_probability,threshold=1e-2):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).
    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """


    v = optimal_value(n_states, n_actions, transition_probability, r,
                      discount, threshold)


    # Get Q using equation 9.2 from Ziebart's thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = transition_probability[i, j, :]
            Q[i, j] = p.dot(r + discount*v)

    Q -= Q.max(axis=1).reshape((n_states, 1))
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q


def find_policy_jit(n_states, r, n_actions, discount,
                           transition_probability,v=None,stochastic=True,threshold=1e-2):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).
    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """


    v = optimal_value_jit(n_states, n_actions, transition_probability, r,
                      discount, threshold)

    # jit speedup
    Q = _iterateQ(n_states, r, v,n_actions, discount, transition_probability)

    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

@njit
def _iterateQ(n_states,r,v, n_actions, discount,
                           transition_probability):

    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = transition_probability[i, j, :]
            Q[i, j] = p.dot(r + discount*v)

    return Q
