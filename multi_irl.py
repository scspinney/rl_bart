from multiprocessing import Process
from utils import load_data
from itertools import product
import time
from maxent import *


def setup_maxent(x):
    return x*x


if __name__ == '__main__':

    maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
    year = 2
    N = 148
    epochs = [10, 50, 100, 250, 500, 1000]
    n_iters = 1
    lrs = [1E-2, 1E-3, 1E-4, 1E-5]
    lr_decay = [1, 5, 10]
    gamma = 1

    feature_matrices, Tprob, trajectories = load_data(maindir, year, N)

    #TODO: hard coded
    num_processes = 4

    # with Pool(num_processes) as p:
    #     for i, (epoch, lr, lr_d) in enumerate(product(epochs,lrs,lr_decay)):
    #         target_process = i % 4

    starttime = time.time()
    processes = []
    for epoch, lr, lr_d in product(epochs,lrs,lr_decay):

        p = Process(target=maxent_irl, args=(maxent_irl(maindir,
                                                        year,
                                                        feature_matrices,
                                                        Tprob,
                                                        gamma,
                                                        trajectories,
                                                        lr,
                                                        lr_d,
                                                        n_iters,
                                                        epoch)))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))
