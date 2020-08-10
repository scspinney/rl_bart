import numpy as np
import glob as glob
import os


def split_fname(s):
     """
     Splits the file name into identifying variables.
     :param s: file name
     :return: tuple of variable name and value
     """

     head = s.rstrip('npy').rstrip('.0123456789')
     tail = s[len(head):]
     # TODO: hard fix for the last term
     tail = tail.rstrip('.npy')
     return head, tail


def load_data(maindir,year,N):


    # trajectories
    traj_paths = os.path.join(maindir, 'data', f'V{year}')
    trajectories = [np.load(p, allow_pickle=True) for p in
                    np.sort(glob.glob(os.path.join(traj_paths, '**', 'traj.npy')))]

    # trajectories = np.load(traj_path,allow_pickle=True)
    # trajectories = trajectories[1:N]

    # feature matrices
    fmat_paths = os.path.join(maindir, 'data', f'V{year}')
    feature_matrices = [np.load(p, allow_pickle=True) for p in
                        np.sort(glob.glob(os.path.join(fmat_paths, '**', 'fmat.npy')))]

    # transition prob: they are all the same so just pick one
    Tprob = np.load(os.path.join(maindir, 'data', f'transition_prob_Y{year}.npy'))

    # TODO: temp
    Tprob = Tprob[:-2, :, :-2]
    Tprob = np.ascontiguousarray(Tprob)

    # TODO: temporary for debugging 12th feature
    trajectories = trajectories[:N]
    feature_matrices = feature_matrices[:N]

    return feature_matrices, Tprob, trajectories


def read_multi_data(maindir,year,kind="theta"):
    files = glob.glob(os.path.join(maindir,f"{kind}*V{year}*N*E*LR*LRD*.npy"))
    data_list = []
    for f in files:
        var_dict = {k:v for k,v in [split_fname(s) for s in f.split("_")]}
        var_dict['fname'] = f
        data_list.append(var_dict)
    return data_list

