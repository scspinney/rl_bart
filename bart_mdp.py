import numpy as np



class BART():

    def __init__(self,N_STATES,N_ACTIONS,N_TRIALS,popPoints):
        self.n_states = N_STATES
        self.n_actions = N_ACTIONS
        self.n_trials = N_TRIALS
        #self.tprob = transition_probability
        self.popPoints = popPoints

        # init starting values
        self.state=0
        self.trial=0
        self.trial_state = 0
        #self.reward = np.zeros(N_STATES)

    def step_forward(self,action):

        if action == 0: # pump

            prob_pop = 1 / (self.n_states + 1 - self.state)

            if np.random.binomial(1,prob_pop,1): # pop

                reward = -10*(self.state-1)
                self.trial_state = 1

            else: # no pop

                reward = 10
                self.trial_state = 0

            self.state += 1 # move forward

        elif action == 1: # save
            reward = 0
            self.trial_state = 1

        else:
            print("Invalid action! 0 or 1 only.")
            return -1

        return reward


    def play(self,agent):

        for t in range(self.n_trials):

            # update
            self.trial = t
            self.state = 0
            # reinitialize
            self.trial_state = 0

            while self.trial_state == 0:

                action = agent.take_action(self.state)
                reward = self.step_forward(action)
                print(f"Action: {action}, Reward: {reward}")
                agent.update_policy((self.state-1),action,reward)

        print("Game finished.")


if __name__ == "__main__":

    from run_rl_agent import rlAgent
    import numpy as np

    N_STATES=128
    N_ACTIONS=2
    N_TRIALS=30
    #transition_probability = np.load('data/transition_prob_Y2.npy')
    popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

    game = BART(N_STATES,N_ACTIONS,N_TRIALS,popPoints)
    agent = rlAgent(type='QL',lr=0.9, epsilon=0.01,discount=0.9,n_states=128,n_actions=2)
    game.play(agent)
    print(agent.policy)
    print("Done playing")