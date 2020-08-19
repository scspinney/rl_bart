import numpy as np
from utils import print_run_params

class BART():

    def __init__(self,N_STATES,N_ACTIONS,N_TRIALS,popPoints):
        self.n_states = N_STATES
        self.n_actions = N_ACTIONS
        self.n_trials = N_TRIALS
        #self.tprob = transition_probability
        self.popPoints = popPoints
        self.cur_traj = []

        # init starting values
        self.state=0
        self.trial=0
        self.trial_state = 0
        #self.reward = np.zeros(N_STATES)

    def step_forward(self,action):

        # update current trajectory
        self.cur_traj.append((self.state, action))

        if action == 0: # pump

            #prob_pop = 1 / (self.n_states + 1 - self.state)

            if self.state+1 == self.popPoints[self.trial]: # pop
            #if np.random.binomial(1,prob_pop,1): # pop

                reward = -10*(self.state-1)
                self.trial_state = -1

            else: # no pop

                reward = 10
                self.trial_state = 0

        elif action == 1: # save
            reward = 0
            self.trial_state = 1

        else:
            print("Invalid action! 0 or 1 only.")
            return -1

        self.state += 1  # move forward

        return reward


    def play(self,agent):

        for t in range(self.n_trials):

            # update
            self.trial = t
            self.state = 0

            # reinitialize
            self.trial_state = 0
            self.cur_traj = []

            while self.trial_state == 0:

                action = agent.take_action(self.state)

                # returns reward and updates state to next state
                reward = self.step_forward(action)


                agent.update_policy((self.state-1),action,reward)

                # update agents state saving
                if self.trial_state == 0:
                    continue
                elif self.trial_state == 1:
                    agent.save_states.append(self.state-1)
                elif self.trial_state == -1:
                    agent.pop_states.append(self.state-1)

            # trajectory ended
            agent.trajectories.append(self.cur_traj)
            not_popped = True if self.trial_state == 1 else False
            agent.update_next_fmat(self.trial,self.state-1,not_popped)



if __name__ == "__main__":

    from run_rl_agent import rlAgent

    params = {
    'N_STATES':128,
    'N_ACTIONS':2,
    'N_TRIALS':30,
    'N_FEAT':11,
    'N_EXPERTS':138,
    }

    #transition_probability = np.load('data/transition_prob_Y2.npy')
    popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

    print_run_params(**params)

    # run a number of RL experts
    fmats=np.zeros((params['N_EXPERTS'],params['N_TRIALS'],params['N_STATES'],params['N_FEAT']))
    trajectories=[]
    for e in range(params['N_EXPERTS']):

        # begin new game of N_TRIALS
        game = BART(params['N_STATES'], params['N_ACTIONS'], params['N_TRIALS'], popPoints)

        # init a new agent
        #agent = rlAgent(type='QL', lr=0.8, epsilon=0.01, discount=0.8, n_states=128, n_actions=2)
        agent = rlAgent(type='Random', lr=0, epsilon=1, discount=0.8, n_states=128, n_actions=2)
        agent.init_feature_matrix(params['N_TRIALS'], params['N_FEAT'])

        # play bart
        game.play(agent)

        # save feature matrices and trajectories
        fmats[e] = agent.fmat
        trajectories.append(agent.trajectories)

    np.save(f"data/agents/feature_matricesN{params['N_EXPERTS']}T{params['N_TRIALS']}S{params['N_STATES']}F{params['N_FEAT']}.npy",fmats,allow_pickle=True)
    np.save(f"data/agents/trajectoriesN{params['N_EXPERTS']}T{params['N_TRIALS']}S{params['N_STATES']}F{params['N_FEAT']}.npy",trajectories,allow_pickle=True)


    print("Done playing")