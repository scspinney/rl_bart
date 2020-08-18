from value_iteration import find_policy_jit
import numpy as np


def likelihood(N_TRIAL,trajectories, feature_matrices, weights, discount, Tprob):



    N_STATES, N_ACTIONS, _  = np.shape(Tprob)
    N_EXPERTS = len(trajectories)


    LL = 0


    # for traj in trajectories:
    #     for b in range(N_TRIAL):
    #         for ind, (state, action) in enumerate(traj[b]):

    for e in range(N_EXPERTS):
        all_expert_trajs = trajectories[e]

        for t, traj in enumerate(all_expert_trajs):

            curr_fmat = feature_matrices[e][t][:-2]

            rewards = np.dot(curr_fmat, weights)

            policy = find_policy_jit(N_STATES, rewards, N_ACTIONS, discount, Tprob)
            ll = 0
            for i, (state,action) in enumerate(traj):

                if i == len(traj)-1:
                    break

                next_state = traj[i+1][0]

                ll += np.log(policy[state,action]) + np.log(Tprob[state,action,next_state])
            #print(f"Trajectory LL for expert of length {len(traj)}: {ll}")
            LL+=ll

    LL = LL / (N_EXPERTS*N_TRIAL)

    return LL




