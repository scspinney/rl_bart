import numpy as np
import os
from value_iteration import *


def maxent_irl(maindir,year,feature_matrices,Tprob, gamma, trajectories, lr, n_iters,use_prior=False):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
    inputs:
      feature_matrix      ExTxNxD matrix - the features for each state
      Tprob         NxAxN_ACTIONS matrix - P_a[s0, a, s1] is the transition prob of
                                         landing at state s1 when taking action
                                         a at state s0
      gamma       float - RL discount factor
      trajs       a list of demonstrations
      lr          float - learning rate
      n_iters     int - number of optimization steps
    returns
      rewards     Nx1 vector - recoverred state rewards
    """
    N_STATES, N_ACTIONS, _  = np.shape(Tprob)
    N_EXPERTS = len(trajectories)
    N_TRIALS = N_EXPERTS*30
    N_FEAT = feature_matrices[0].shape[-1]


    # init parameters
    if use_prior and year > 1:
        theta = np.load(os.path.join(maindir,'data','results',f'rewards_weights_{year-1}.npy'))

    else:
        theta = np.random.uniform(size=(N_FEAT,))

    # keeping track of gradients
    gradients = np.zeros((n_iters,N_FEAT))
    policy = np.zeros((N_STATES, N_ACTIONS))

    # observed state visitation frequency
    # for t in range(N_TRIALS):
    #     for s in range(len(trajectories[t])):
    #         svf[trajectories[t][s][0]]+=1
    #
    # svf /= N_TRIALS

    for epoch in range(round(n_iters / 2)):
        print(f"Epochs {epoch/round(n_iters / 2)} completed.")
        for e in range(N_EXPERTS):
            all_expert_trajs = trajectories[e]
            # shuffle indices
            ind = np.random.permutation(range(len(all_expert_trajs)))

            all_expert_trajs = all_expert_trajs[ind]
            feature_matrix = feature_matrices[e][ind]
            for t,trajectory in enumerate(all_expert_trajs):

                curr_fmat = feature_matrix[t] # this traj feature matrix

                # calc feature expectations
                feat_exp = np.zeros([N_FEAT])
                svf = np.zeros(N_STATES)


                for state, _, _ in trajectory:
                    feat_exp += curr_fmat[state]

                #feat_exp /= N_TRIALS

                # optimization
                for iteration in range(n_iters):

                    #if iteration % (n_iters / 20) == 0:
                     #   print(f"Epoch {epoch}, iteration: {iteration/round(n_iters / 2)} completed.")

                    # compute expected reward for summarized feature matrix for every state
                    rewards = np.dot(curr_fmat, theta)

                    #_, policy = optimal_policy(Tprob, rewards, gamma, error=0.1)
                    policy = find_policy(N_STATES, rewards, N_ACTIONS, gamma, Tprob)
                    #print(f"Policy: {policy}")

                    # compute expected state visitation frequencies
                    esvf = compute_state_visition_freq(Tprob, gamma, trajectory, policy)
                    #print(f"SVF: {svf}")

                    # compute gradients
                    #grad = feat_exp - esvf.dot(feature_matrix)
                    grad = feat_exp - esvf.dot(curr_fmat)
                    gradients[iteration,] = grad
                    print(f"Grad sum : {np.sum(grad)}")

                    # update params
                    theta = theta - lr * grad
                    #print(f"Theta : {theta}")

    # rewards over states
    #TODO: normalize
    #rewards = np.dot(feature_matrix, theta)
    np.save(f'/Users/sean/Projects/rl_bart/data/results/policy_V{year}.npy',policy,allow_pickle=True)
    np.save(f'/Users/sean/Projects/rl_bart/data/results/esvf_V{year}.npy',esvf, allow_pickle=True)
    #np.save(f'/Users/sean/Projects/rl_bart/data/results/svf_V{year}.npy', svf, allow_pickle=True)
    np.save(f'/Users/sean/Projects/rl_bart/data/results/gradients_V{year}.npy', gradients, allow_pickle=True)



    return theta




# def maxent_irl(maindir,year,feature_matrix,Tprob, gamma, trajectories, lr, n_iters,use_prior=False):
#     """
#     Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
#     inputs:
#       feature_matrix      ExTxNxD matrix - the features for each state
#       Tprob         NxAxN_ACTIONS matrix - P_a[s0, a, s1] is the transition prob of
#                                          landing at state s1 when taking action
#                                          a at state s0
#       gamma       float - RL discount factor
#       trajs       a list of demonstrations
#       lr          float - learning rate
#       n_iters     int - number of optimization steps
#     returns
#       rewards     Nx1 vector - recoverred state rewards
#     """
#     N_STATES, N_ACTIONS, _  = np.shape(Tprob)
#     N_TRIALS = len(trajectories)
#     N_FEAT = feature_matrix.shape[-1]
#
#     # init parameters
#     if use_prior and year > 1:
#         theta = np.load(os.path.join(maindir,'data','results',f'rewards_weights_{year-1}.npy'))
#
#     else:
#         theta = np.random.uniform(size=(N_FEAT,))
#
#     # keeping track of gradients
#     gradients = np.zeros((n_iters,N_FEAT))
#
#     # calc feature expectations
#     feat_exp = np.zeros([N_FEAT])
#     svf = np.zeros(N_STATES)
#
#     # observed state visitation frequency
#     for t in range(N_TRIALS):
#         for s in range(len(trajectories[t])):
#             svf[trajectories[t][s][0]]+=1
#
#     svf /= N_TRIALS
#
#     #TODO: validate
#     for trajectory in trajectories:
#         for state, _, _ in trajectory:
#             feat_exp += feature_matrix[state]
#
#     feat_exp /= N_TRIALS
#
#     policy = np.zeros((N_STATES,N_ACTIONS))
#
#     # optimization
#     for iteration in range(n_iters):
#
#         if iteration % (n_iters / 20) == 0:
#             print(f"iteration: {iteration/n_iters}")
#
#         # compute expected reward for summarized feature matrix for every state
#         rewards = np.dot(feature_matrix, theta)
#
#         _, policy = optimal_policy(Tprob, rewards, gamma, error=0.1)
#         #print(f"Policy: {policy}")
#
#         # compute expected state visitation frequencies
#         esvf = compute_state_visition_freq(Tprob, gamma, trajectories, policy)
#         #print(f"SVF: {svf}")
#
#         # compute gradients
#         #grad = feat_exp - esvf.dot(feature_matrix)
#         grad = feat_exp - feature_matrix.T.dot(esvf)
#         gradients[iteration,] = grad
#         print(f"Grad sum : {np.sum(grad)}")
#
#         # update params
#         theta = theta + lr * grad
#         print(f"Theta : {theta}")
#
#     # rewards over states
#     #TODO: normalize
#     #rewards = np.dot(feature_matrix, theta)
#     np.save(f'/Users/sean/Projects/rl_bart/data/results/policy_V{year}.npy',policy,allow_pickle=True)
#     np.save(f'/Users/sean/Projects/rl_bart/data/results/esvf_V{year}.npy',esvf, allow_pickle=True)
#     np.save(f'/Users/sean/Projects/rl_bart/data/results/svf_V{year}.npy', svf, allow_pickle=True)
#     np.save(f'/Users/sean/Projects/rl_bart/data/results/gradients_V{year}.npy', gradients, allow_pickle=True)
#
#
#     return theta
