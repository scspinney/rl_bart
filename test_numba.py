from numba import njit
from value_iteration import optimal_value, optimal_value_jit, compute_state_visition_freq, compute_state_visition_freq_jit, find_policy, find_policy_jit
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

def repeat_find_policy(N):

    for _ in range(N):
        find_policy(N_STATES, rewards, N_ACTIONS, 0.99, tprob)


def repeat_find_policy_jit(N):

    for _ in range(N):
        find_policy_jit(N_STATES, rewards, N_ACTIONS, 0.99, tprob)


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

N = [10,20,30,40,80,100,120,200,300,400]

tprob = np.ascontiguousarray(tprob)

speedups = {'time':[],
            'function':[],
            'numba':[],
            'N':[]
            }

for n in N:

    # ### Begin test suite
    #
    # print(f"\nTesting Numba Speedup on optimal policy search with N={n}:\n")
    # t = timeit.Timer(lambda: repeat_optimal_value(n))
    #
    # #update
    # speedups['N'].append(n)
    # speedups['time'].append(t.timeit(number=1))
    # speedups['function'].append('optimal_policy')
    # speedups['numba'].append(False)
    #
    # print (f"optimal policy run {n} times without numba --- {t.timeit(number=1):.2f} seconds ---")
    #
    # t = timeit.Timer(lambda: repeat_optimal_valu_jit(n))
    #
    # # update
    # speedups['N'].append(n)
    # speedups['time'].append(t.timeit(number=1))
    # speedups['function'].append('optimal_policy')
    # speedups['numba'].append(True)
    #
    # print (f"optimal policy run {n} times with numba --- {t.timeit(number=1):.2f} seconds ---")
    #
    # ### New Test
    #
    # print(f"\nTesting Numba Speedup on optimal state visitation frequency with N={n}:\n")
    # t = timeit.Timer(lambda: repeat_esvf(n))
    #
    # # update
    # speedups['N'].append(n)
    # speedups['time'].append(t.timeit(number=1))
    # speedups['function'].append('esvf')
    # speedups['numba'].append(False)
    #
    # print (f"State visitation frequency algorithm run {n} times without numba --- {t.timeit(number=1):.2f} seconds ---")
    #
    # t = timeit.Timer(lambda: repeat_esvf_jit(n))
    #
    # # update
    # speedups['N'].append(n)
    # speedups['time'].append(t.timeit(number=1))
    # speedups['function'].append('esvf')
    # speedups['numba'].append(True)
    #
    # print (f"State visitation frequency algorithm run {n} times with numba --- {t.timeit(number=1):.2f} seconds ---")

    ### New Test

    print(f"\nTesting Numba Speedup on find_policy with N={n}:\n")
    t = timeit.Timer(lambda: repeat_find_policy(n))

    # update
    speedups['N'].append(n)
    speedups['time'].append(t.timeit(number=1))
    speedups['function'].append('find_policy')
    speedups['numba'].append(False)

    print (f"find_policy function run {n} times without numba --- {t.timeit(number=1):.2f} seconds ---")

    t = timeit.Timer(lambda: repeat_find_policy_jit(n))

    # update
    speedups['N'].append(n)
    speedups['time'].append(t.timeit(number=1))
    speedups['function'].append('find_policy')
    speedups['numba'].append(True)

    print (f"find_policy function run {n} times with numba --- {t.timeit(number=1):.2f} seconds ---")





results = pd.DataFrame().from_dict(speedups)

g=sns.FacetGrid(col="function",hue="numba",sharex=True, sharey=False, data=results)
g=(g.map(sns.lineplot,"N","time").add_legend())
g
plt.show()

plt.savefig(f'results/numba_speedup_tests.png')



