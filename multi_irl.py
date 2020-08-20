from multiprocessing import Process
from utils import load_data
from itertools import product
import time
import pickle
from maxent import *


def setup_maxent(x):
    return x*x


if __name__ == '__main__':

    maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
    year = 2
    N = 138
    epochs = [20]
    n_iters = 1
    lrs = [1E-4]
    lr_decay = [1]
    random_seeds = [0,42,100,333,666]
    gamma = 1
    ai=False
    ai_type=''

    feature_matrices, Tprob, trajectories = load_data(maindir, year, N)

    #TODO: hard coded
    num_processes = 5

    # with Pool(num_processes) as p:
    #     for i, (epoch, lr, lr_d) in enumerate(product(epochs,lrs,lr_decay)):
    #         target_process = i % 4

    starttime = time.time()
    processes = []
    for epoch, lr, lr_d, seed in product(epochs,lrs,lr_decay,random_seeds):

        p = Process(target=maxent_irl, args=(maindir,
                                            N,
                                            year,
                                            feature_matrices,
                                            Tprob,
                                            gamma,
                                            trajectories,
                                            lr,
                                            lr_d,
                                            n_iters,
                                            epoch,
                                            seed,
                                            ai,
                                            ai_type))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    # NO saving needed
    #with open('results/multi_irl_results.pickle', 'wb') as output:
     #   for i, process in enumerate(processes):
      #      pickle.dump(process, output, pickle.HIGHEST_PROTOCOL)

    print('That took {} seconds'.format(time.time() - starttime))
