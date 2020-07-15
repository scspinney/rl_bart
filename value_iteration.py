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


def compute_state_visition_freq(N,N_STATES,N_ACTIONS,Tprob, gamma, trajectory, policy,popPoint):
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


    #trajs = np.reshape(trajs,-1)
    trajectory_length = len(trajectory)

    start_state_count = np.zeros(N_STATES)
    start_state_count[0] = 1
    #expected_svf = start_state_count
    expected_svf = np.zeros(N_STATES)

    while True:

        tmp_exp_svf = start_state_count.copy()
        for s in range(N_STATES-2):

            # pump no pop
            tmp_exp_svf[s+1] = tmp_exp_svf[s+1] + expected_svf[s]*policy[s,0]*Tprob[s,0,s+1]

            # pump and pop
            tmp_exp_svf[-1] = tmp_exp_svf[-1] + expected_svf[s]*policy[s, 0]*Tprob[s,0,-1]

            # save
            tmp_exp_svf[-2] = tmp_exp_svf[-2] + expected_svf[s]*policy[s, 1] * Tprob[s,1,-2]

        if all(abs(tmp_exp_svf - expected_svf) < 0.1):
            break
        else:
            expected_svf = tmp_exp_svf

    return expected_svf



def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
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

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    #V = np.zeros((n_states, 1))
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))


    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states-2):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))
                # terminal state
                new_V[-2:] = 0

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q



