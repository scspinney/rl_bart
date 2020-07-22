from value_iteration import find_policy
import numpy as np


def likelihood(trajectories, feature_matrices, weights, discount, Tprob):



    N_STATES, N_ACTIONS, _  = np.shape(Tprob)
    N_EXPERTS = len(trajectories)



    for e in range(N_EXPERTS):
        all_expert_trajs = trajectories[e]

        for t, traj in enumerate(all_expert_trajs):

            curr_fmat = feature_matrices[e][t]

            rewards = rewards = np.dot(curr_fmat, weights)

            policy = find_policy(N_STATES, rewards, N_ACTIONS, discount, Tprob)
            ll = 0
            for i, (state,action, _) in enumerate(traj[:-1]):

                if i == len(traj)-1:
                    break

                next_state = traj[i+1][0]

                ll += np.log(policy[state,action]) + np.log(Tprob[state,action,next_state])
            print(f"Trajectory LL for expert of length {len(traj)}: {ll}")

    return ll




