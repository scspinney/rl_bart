from numba import njit
from value_iteration import optimal_value, optimal_value_jit, compute_state_visition_freq, compute_state_visition_freq_jit
import timeit
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def repeat_optimal_value(N):

    for _ in range(N):
        optimal_value(N_STATES, N_ACTIONS, tprob, rewards, 0.99)

@njit
def repeat_optimal_valu_jit(N):

    for _ in range(N):
        optimal_value_jit(N_STATES, N_ACTIONS, tprob, rewards, 0.99)


def repeat_esvf(N):

    for _ in range(N):
        compute_state_visition_freq(N_STATES,tprob,policy)


@njit
def repeat_esvf_jit(N):

    for _ in range(N):
        compute_state_visition_freq_jit(N_STATES,tprob,policy)

# get an example of feature matrix

fmat = np.load('data/V2/001/fmat.npy')
tprob = np.load('data/V2/001/transition_prob.npy')
policy = np.load('results/policy_V2.npy')

tprob = tprob[:-2,:,:-2]
fmat = fmat[20,:-2,:]

N_STATES, N_ACTIONS, _ = np.shape(tprob)
N_TRIALS = 30
_,N_FEAT = np.shape(fmat)

theta = np.random.uniform(size=(N_FEAT,))

rewards = np.dot(fmat, theta)

N = [10,100,1000,10000,20000]

tprob = np.ascontiguousarray(tprob)

speedups = {'time':[],
            'function':[],
            'numba':[],
            'N':[]
            }

for n in N:



    ### Begin test suite
    print(f"\nTesting Numba Speedup on optimal policy search with N={n}:\n")
    t = timeit.Timer(lambda: repeat_optimal_value(n))

    #update
    speedups['N'].append(n)
    speedups['time'].append(t.timeit(number=1))
    speedups['function'].append('optimal_policy')
    speedups['numba'].append(False)

    print (f"optimal policy run {n} times without numba --- {t.timeit(number=1):.2f} seconds ---")

    t = timeit.Timer(lambda: repeat_optimal_valu_jit(n))

    # update
    speedups['N'].append(n)
    speedups['time'].append(t.timeit(number=1))
    speedups['function'].append('optimal_policy')
    speedups['numba'].append(True)

    print (f"optimal policy run {n} times with numba --- {t.timeit(number=1):.2f} seconds ---")

    ###

    print(f"\nTesting Numba Speedup on optimal state visitation frequency with N={n}:\n")
    t = timeit.Timer(lambda: repeat_esvf(n))

    # update
    speedups['N'].append(n)
    speedups['time'].append(t.timeit(number=1))
    speedups['function'].append('esvf')
    speedups['numba'].append(False)

    print (f"State visitation frequency algorithm run {n} times without numba --- {t.timeit(number=1):.2f} seconds ---")

    t = timeit.Timer(lambda: repeat_esvf_jit(n))

    # update
    speedups['N'].append(n)
    speedups['time'].append(t.timeit(number=1))
    speedups['function'].append('esvf')
    speedups['numba'].append(True)

    print (f"State visitation frequency algorithm run {n} times with numba --- {t.timeit(number=1):.2f} seconds ---")


results = pd.DataFrame().from_dict(speedups)

g=sns.FacetGrid(col="function",hue="numba",sharex=True, sharey=False, data=results)
g=(g.map(sns.lineplot,"N","time").add_legend())
g
plt.show()

plt.savefig(f'results/numba_speedup_tests.png')



