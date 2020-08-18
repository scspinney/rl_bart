from maxent import *
from plots import plot_rl_trajectories
from value_iteration import find_policy
from utils import *
import pickle
import datetime
import random



class irlAgent():

    # Features:
    # 1: # of times being in this state
    # 2: whether this state was burst in previous trial
    # 3: whether this state was save in previous trial
    # 4: whether this state was burst in 2nd previous trial
    # 5: whether this state was save in 2nd previous trial
    # 6: whether this state was burst in 3rd previous trial
    # 7: whether this state was save in 3rd previous trial
    # 8: whether is average burst status
    # 9: whether is average save status
    # 10: whether is average end status
    # 11: # of steps (pumps) in current trial

    EPSILON_DEFAULT = 0.3

    def __init__(self,weights, gamma, fmat, risk=0):
        self.weights = weights
        self.gamma = gamma
        self.fmat = fmat
        self.save_states = []
        self.pop_states = []
        self.trajectories = []

        #TODO: risky behavior injection
        self.risk = risk


    def update_saves(self,state):
        self.save_states.append(state)

    def update_pops(self,state):
        self.pop_states.append(state)

    def update_next_fmat(self,t,end_state,not_popped):

        # mod F1
        self.fmat[t+1,:,0] = np.array(list((self.fmat[t,:end_state+1,0] + 1)) + list(self.fmat[t,end_state+1:,0]))

        if not_popped == False:

            # mod F2
            self.fmat[t+1,end_state,1] = 1

            # mod F8
            avg_pop_index = int(np.mean(self.pop_states))
            self.fmat[t+1,avg_pop_index,7] = 1

        else:

            # mod F3
            self.fmat[t + 1, end_state, 2] = 1

            # mod F9
            avg_save_index = int(np.mean(self.save_states))
            self.fmat[t+1,avg_save_index,8] = 1

        # mod F3
        self.fmat[t + 1, end_state, 3] = 1 if self.fmat[t - 1, end_state, 3] else 0

        # mod F4
        self.fmat[t + 1, end_state, 4] = 1 if self.fmat[t - 1, end_state, 4] else 0

        # mod F5
        self.fmat[t + 1, end_state, 3] = 1 if self.fmat[t - 2, end_state, 3] else 0

        # mod F6
        self.fmat[t + 1, end_state, 4] = 1 if self.fmat[t - 2, end_state, 4] else 0

        # mod F10
        avg_end_index = int(np.mean(self.save_states+self.pop_states))
        self.fmat[t + 1, avg_end_index, 9] = 1
        print(self.fmat[t+1][:10])
        return self.fmat[t+1]

    def take_action(self,state,policy,decay=False):

        #TODO: take risky action
        if np.random.rand() < self.risk:
            action = 0
        else:
            action = 1 if np.random.binomial(1,policy[state,1],1) else 0

        if decay:
            self.risk /= (self.risk+1)

        return action


def generate_trajectories(N_EXPERTS,N_TRIALS,N_FEAT,N_STATES,Tprob,weights,gamma,popPoints):

    trajectories = []
    # feature matrix
    fmat = np.zeros((N_EXPERTS,N_TRIALS,N_STATES, N_FEAT))

    # init the features that aren't dynamic i.e. the pump#
    fmat[:,:,:,-1] = np.arange(N_STATES)

    for e in range(N_EXPERTS):

        # init an agent
        agent = irlAgent(weights, gamma, fmat[e],risk=0)

        for t in range(1,N_TRIALS-1):
            print(f"Trial {t}")
            # reinitialize trial flags
            not_pop = True
            not_save = True
            state = 0  # first pump
            traj=[]

            while not_pop and not_save:

                # estimated reward for all state as defined by current fmat and weights
                reward = np.dot(agent.fmat[t], weights) # (N_STATES,)

                # get policy using reward
                policy = find_policy_jit(N_STATES,reward,2,gamma,Tprob)

                # make choice
                choice = agent.take_action(state,policy)

                # update trajectories
                traj.append((state,choice))

                if choice == 0: # pump

                    if state == popPoints[t]:
                        not_pop = False
                        agent.update_pops(state)
                    else:
                        state+=1

                else: # save
                    not_save = False
                    agent.update_saves(state)

            agent.trajectories.append(traj)
            fmat[e,t+1] = agent.update_next_fmat(t,state,not_pop)
        trajectories.append(agent.trajectories)


    np.save(f'data/agents/feature_matricesN{N_EXPERTS}T{N_TRIALS}S{N_STATES}F{N_FEAT}.npy',fmat)
    np.save(f'data/agents/trajectoriesN{N_EXPERTS}T{N_TRIALS}S{N_STATES}F{N_FEAT}.npy',trajectories,allow_pickle=True)


def generate_trajectories_human_features(N_EXPERTS,N_TRIALS,N_FEAT,N_STATES,Tprob,weights,gamma,popPoints,human_fmat_path):

    trajectories = []
    # feature matrix
    fmat = np.load(human_fmat_path)

    for e in range(N_EXPERTS):

        # init an agent
        agent = irlAgent(weights, gamma, fmat[e],risk=0)

        for t in range(1,N_TRIALS-1):
            print(f"Trial {t}")
            # reinitialize trial flags
            not_pop = True
            not_save = True
            state = 0  # first pump
            traj=[]

            while not_pop and not_save:

                # estimated reward for all state as defined by current fmat and weights
                reward = np.dot(agent.fmat[t], weights) # (N_STATES,)

                # get policy using reward
                policy = find_policy_jit(N_STATES,reward,2,gamma,Tprob)

                # make choice
                choice = agent.take_action(state,policy)

                # update trajectories
                traj.append((state,choice))

                if choice == 0: # pump

                    if state == popPoints[t]:
                        not_pop = False
                        agent.update_pops(state)
                    else:
                        state+=1

                else: # save
                    not_save = False
                    agent.update_saves(state)

            agent.trajectories.append(traj)
        trajectories.append(agent.trajectories)


    np.save(f'data/agents/feature_matricesN{N_EXPERTS}T{N_TRIALS}S{N_STATES}F{N_FEAT}.npy',fmat)
    np.save(f'data/agents/trajectoriesN{N_EXPERTS}T{N_TRIALS}S{N_STATES}F{N_FEAT}.npy',trajectories,allow_pickle=True)



def recover_trajectories(N_EXPERTS,N_STATES,N_ACTIONS,Tprob,weights,gamma,trajectories,feature_matrices):
    l = []
    for e in range(N_EXPERTS):

        all_expert_trajs = trajectories[e]
        feature_matrix = feature_matrices[e]

        for t, trajectory in enumerate(all_expert_trajs):

            curr_fmat = feature_matrix[t][:-2, :]  # this traj feature matrix

            liu_rewards = np.dot(curr_fmat, weights['Liu'])
            our_rewards = np.dot(curr_fmat, weights['Ours'])

            liu_policy = find_policy(N_STATES, liu_rewards, N_ACTIONS, gamma, Tprob)
            our_policy = find_policy(N_STATES, our_rewards, N_ACTIONS, gamma, Tprob)

            svf = get_svf( N_STATES, trajectory)

            liu_esvf = compute_state_visition_freq_jit(N_STATES, Tprob, liu_policy)
            our_esvf = compute_state_visition_freq_jit(N_STATES, Tprob, our_policy)

            l.append({'liu_esvf': liu_esvf,
                      'our_esvf': our_esvf,
                      'svf': svf,
                      'balloon': t,
                      'expert': e})
    return l


###### INITIALIZATION STUFF

maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
year=2
N=138


popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

feature_matrices, Tprob, trajectories = load_data(maindir,year,N)

#N_EXPERTS = len(feature_matrices)
#N_TRIAL, N_STATES, N_FEAT = np.shape(feature_matrices[0])
N_EXPERTS = 5
N_TRIAL = 30
N_STATES = 128
N_FEAT = 11
#N_STATES -= 2
N_ACTIONS=2

gamma=1

weights = {'Liu': [5, 0.3,   0.7,  0.7, 1.2, 0.1, 1.2, 0.6, 1.35, 1, 2.5],

             'Ours': [0.49049365,  0.97522182,  0.81013244,  0.59797393,  0.16992047,
                      0.15576953,  0.06382926,  0.87656816,  0.60628161,  0.71614336,
                      -0.044085]

           #'Ours': [0.36168188, 1.27113523, 1.74129503, 0.59647852, 0.32134919,
            #        0.15810741, 0.11944254, 1.04198014, 0.64163186, 0.81853915, 0.04663256]
           }

w = np.load('results/theta_V2_N138_E500_LR0.0001_LRD1_S42.npy')
weights['Ours'] = w[-1].mean(axis=(0,1))

# popPoints resampled to match trial number
popPoints = random.choices(popPoints,k=N_TRIAL)
#generate_trajectories(N_EXPERTS,N_TRIAL,N_FEAT,N_STATES,Tprob,weights['Ours'],gamma,popPoints)

###### END OF INITIALIZATION STUFF


out_name = f'results/rl_esvf{str(datetime.date.today())}.pl'
clobber=True
if not os.path.exists(out_name) or clobber:

    trajectories = recover_trajectories(N_EXPERTS,N_STATES,N_ACTIONS,Tprob,weights,gamma,trajectories,feature_matrices)
    plot_rl_trajectories(trajectories)
    pickle.dump(trajectories, open(out_name, "wb"))

else:
    trajectories = pickle.load(open(out_name, "rb"))


#plot_rl_trajectories(trajectories)

print("Done.")








