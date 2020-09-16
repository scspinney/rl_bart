import numpy as np
import os
from value_iteration import *
from optimization import Adam



def maxent_irl(maindir,N,year,feature_matrices,Tprob, gamma, trajectories, lr,lr_decay,n_iters,n_epochs,seed,ai,ai_type,method,shuffle_training=True,use_prior=False):
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

    # optimization
    optim = Adam()


    # filename prefix
    suffix = f"V{year}_N{N}_E{n_epochs}_LR{lr}_LRD{lr_decay}_S{seed}"

    #N_STATES-=2

    # init parameters
    if use_prior and year > 1:
        theta = np.load(os.path.join(maindir,'data','results',f'rewards_weights_{year-1}.npy'))

    else:
        # set the random seed
        #np.random.seed(seed)
        theta = np.random.uniform(size=(N_FEAT,))
        #theta = np.zeros((N_FEAT,))

    # keeping track of gradients
    gradients = np.zeros((n_epochs,N_EXPERTS,N_TRIALS,n_iters,N_FEAT))
    theta_vec = np.zeros((n_epochs,N_EXPERTS,N_TRIALS,N_FEAT))
    #policy = np.zeros((N_STATES, N_ACTIONS))
    counter=1
    for epoch in range(n_epochs):

        print(f"Theta: {theta}")
        lr_decay+=1
        for e in range(N_EXPERTS):

            all_expert_trajs = trajectories[e]
            # shuffle trajectories by default
            if shuffle_training:

                ind = np.random.permutation(range(len(all_expert_trajs)))
                all_expert_trajs = all_expert_trajs[ind]
                feature_matrix = feature_matrices[e][ind]

            else:
                feature_matrix = feature_matrices[e]

            for t,trajectory in enumerate(all_expert_trajs):
                #print(f"-------------------- NEW TRAJECTORY NUMBER {t} --------------------------")
                if ai:
                    curr_fmat = feature_matrix[t]
                else:
                    #curr_fmat = feature_matrix[t][:-2,:] # this traj feature matrix
                    curr_fmat = feature_matrix[t]
                feat_exp = np.zeros([N_FEAT])

                for state, _ in trajectory:
                   feat_exp += curr_fmat[state]


                for iteration in range(n_iters):

                    # compute expected reward for feature matrix
                    rewards = np.dot(curr_fmat, theta)
                    #print(rewards)
                    # generate policy
                    policy = find_policy_jit(N_STATES, rewards, N_ACTIONS, gamma, Tprob,method)

                    # get ESVF
                    esvf = compute_state_visition_freq_jit(N_STATES,Tprob,policy)

                    # compute gradients
                    grad = feat_exp - esvf.dot(curr_fmat)
                    #if iteration % 100 == 0: print(f"Grad sum: {np.sum(grad)}")

                    # update weights
                    theta = optim.backwards(theta,grad)
                    #theta += lr/lr_decay * grad
                    #theta += lr * grad

                    gradients[epoch, e, t,iteration, :] = grad
                    
                    if counter % 500 == 0:
                        print(f"Progress at {100*(counter/(n_iters*N_TRIALS*n_epochs*N_EXPERTS)):.2f}% complete...")
                    counter+=1

                theta_vec[epoch, e, t, :] = theta


    print(f"Progress at {100*(counter/(n_iters*N_TRIALS*n_epochs*N_EXPERTS)):.2f}% complete. Saving policy, esvf, weights, and gradients in results.")

    prefix=f"{ai_type}_ai_" if ai else ""

    #np.save(f'results/{prefix}policy_{suffix}.npy',policy,allow_pickle=True)
    #np.save(f'results/{prefix}esvf_{suffix}.npy',esvf, allow_pickle=True)
    np.save(f'results/{prefix}theta_{suffix}.npy', theta_vec, allow_pickle=True)
    np.save(f'results/{prefix}gradients_{suffix}.npy', gradients, allow_pickle=True)

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
                        #TODO: Artifical value increase at saving
                        #rewards[state] += 200
                else:
                    rewards[state] += 10

    rewards = np.array(list(rewards.values()))
    avg_rewards = rewards / (N_EXPERTS*N_TRIAL)

    avg_save_state = np.mean(save_states)

    return avg_rewards, avg_save_state


def get_svf(N_STATES,trajectory):

    svf = np.zeros((N_STATES,))

    for state,action in trajectory:
        svf[state]+=1

    return svf
