from utils import *
from maxent import get_stats
from likelihood import likelihood
from plots import *
import numpy as np

np.set_printoptions(suppress=True)



def ll_table(maindir,year,N_TRIAL,trajectories, feature_matrices,discount, Tprob,clobber=True):

    outname='results/ll_table.csv'
    if os.path.exists(outname) and not clobber:
        return pd.read_csv(outname)

    df = pd.DataFrame().from_dict(read_multi_data(maindir,year))
    fnames = df.fname.values
    ll_list = []
    for f in fnames:
        weights = np.load(f) # last updated weights
        weights = weights[-1].mean(axis=(0,1))
        ll_list.append(likelihood(N_TRIAL,trajectories, feature_matrices, weights, discount, Tprob))
    df['LL'] = pd.Series(ll_list)
    df.sort_values(by="LL",ascending=False,inplace=True)
    df = df.rename(columns={"V": "Year", "E": "Epochs", "N": "Number of experts", "LR":"LR", "LRD": "LR Decay","S":"Seed","LL":"LogLikelihood" })
    df.to_csv('results/ll_table.csv')
    df.iloc[:,1:].to_html('results/ll_table.html',index=False)
    return df



maindir = '/Users/sean/Projects/rl_bart/'
year=2
N=10


## Load data

feature_matrices, Tprob, trajectories = load_data(maindir,year)

N_EXPERTS = len(feature_matrices)
N_TRIAL, N_STATES, N_FEAT = np.shape(feature_matrices[0])
N_STATES -= 2

gradients = np.load(f'results/gradients_V2_N100_E100_LR0.1_LRD1_S100.npy')
weights = np.load('results/theta_V2_N100_E100_LR0.1_LRD1_S100.npy')

#print(weights.shape)
# not really the avg, just last update
avg_weights = weights[-1].mean(axis=(0,1))

obs_exp_rewards, avg_save_state = get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories)

### PLOTTING ###

# multi run results 
#run_data=read_multi_data('results',year,kind="theta")
#plot_multi_weights(run_data)

#plot_reward_landscape('human',N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,avg_weights,feature_matrices,obs_exp_rewards,avg_save_state,'line',clobber=True)
plot_gradients(gradients)
plot_weights(weights,'line')

#avg_LL = likelihood(N_TRIAL,trajectories, feature_matrices, avg_weights, discount=1, Tprob=Tprob)

#lldf=ll_table('results',year,N_TRIAL,trajectories, feature_matrices,1, Tprob)
#plot_ll(lldf)

#print(f"Average Log Likelihood on training demonstrations: N = {N_EXPERTS*N_TRIAL} demonstrations, LL = {avg_LL}")


