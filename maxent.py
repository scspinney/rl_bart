import numpy as np
import os
from value_iteration import *



def maxent_irl(maindir,year,feature_matrices,Tprob, gamma, trajectories, lr,lr_decay,n_iters,n_epochs,popPoints,use_prior=False):
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
    N_TRIALS = 30
    N_FEAT = feature_matrices[0].shape[-1]

    #N_STATES-=2

    # init parameters
    if use_prior and year > 1:
        theta = np.load(os.path.join(maindir,'data','results',f'rewards_weights_{year-1}.npy'))

    else:
        theta = np.random.uniform(size=(N_FEAT,))

    # keeping track of gradients
    gradients = np.zeros((n_epochs,N_EXPERTS,N_TRIALS,N_FEAT))
    policy = np.zeros((N_STATES, N_ACTIONS))

    #for epoch in range(1):
    for epoch in range(round(n_epochs)):

        print(f"Progress {epoch/round(n_iters)}\% completed.")
        print(f"Theta: {theta}")
        for e in range(N_EXPERTS):

            #print(f"Theta for expert {e} on {N_EXPERTS}: {theta}")
            all_expert_trajs = trajectories[e]

            ## shuffle trajectories
            #ind = np.random.permutation(range(len(all_expert_trajs)))
            #all_expert_trajs = all_expert_trajs[ind]
            #feature_matrix = feature_matrices[e][ind]
            feature_matrix = feature_matrices[e]

            for t,trajectory in enumerate(all_expert_trajs):
                print(f"-------------------- NEW TRAJECTORY NUMBER {t} --------------------------")

                curr_fmat = feature_matrix[t][:-2,:] # this traj feature matrix


                # calc feature expectations
                feat_exp = np.zeros([N_FEAT])
                #svf = np.zeros(N_STATES)

                for state, _ in trajectory:
                   feat_exp += curr_fmat[state]

                #feat_exp /= N_TRIALS

                # optimization
                lr_decay = 1
                #while True:
                #for iteration in range(round(n_iters)):
                for iteration in range(n_iters):

                    #if iteration % (n_iters / 20) == 0:
                     #   print(f"Epoch {epoch}, iteration: {iteration/round(n_iters / 2)} completed.")

                    # compute expected reward for summarized feature matrix for every state
                    rewards = np.dot(curr_fmat, theta)

                    #_, policy = optimal_policy(Tprob, rewards, gamma, error=0.1)
                    policy = find_policy(N_STATES, rewards, N_ACTIONS, gamma, Tprob)
                    #print(f"Policy: {policy}")

                    # compute expected state visitation frequencies

                    esvf = compute_state_visition_freq(N_STATES,N_ACTIONS,Tprob, gamma, trajectory, policy)
                    #print(f"SVF: {svf}")

                    # compute gradients
                    grad = feat_exp - esvf.dot(curr_fmat)


                    #gradients[iteration,] = gradgit
                    print(f"Grad sum: {np.sum(grad)}")

                    # update params
                    theta += lr/lr_decay * grad
                    lr_decay+=1

                    if abs(grad.sum()) < 4:
                        # stop from climbing out of min
                        break

                gradients[epoch,e,t,:] = grad



                print(f"Theta: {theta}")
    np.save(f'results/policy_V{year}.npy',policy,allow_pickle=True)
    np.save(f'results/esvf_V{year}.npy',esvf, allow_pickle=True)
    np.save(f'results/theta_V{year}.npy', theta, allow_pickle=True)
    np.save(f'results/gradients_V{year}.npy', gradients, allow_pickle=True)

    return theta

def get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories):
    """

    :param N_EXPERTS:
    :param N_TRIAL:
    :param N_STATES:
    :param N_FEAT:
    :param trajectories:
    :return: (average reward for every state float, average save state int)

    """

    rewards = {i:0 for i in range(N_STATES)}
    svf = np.ones((N_STATES,))
    save_states=[]

    for traj in trajectories:
        for b in range(30):
            for ind, (state, action) in enumerate(traj[b]):
                #update svf
                svf[state] +=1

                if ind == len(traj[b])-1:
                    if action == 0:
                        rewards[state] += -10*(state-1)
                    elif action == 1:
                        save_states.append(state)
                else:
                    rewards[state] += 10

    rewards = np.array(list(rewards.values()))
    avg_rewards = rewards / (N_EXPERTS*N_TRIAL)

    avg_save_state = np.mean(save_states)

    return avg_rewards, avg_save_state


