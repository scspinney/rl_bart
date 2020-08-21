import numpy as np
import glob as glob
import pandas as pd
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


def split_fname2(s):
    """
    Splits the file name into identifying variables.
    :param s: file name
    :return: tuple of variable name and value
    """
    var_dict = dict()
    vars = s.split("_")
    var_dict["fname"] = s
    var_dict["V"] = vars[1][1]
    var_dict["N"] = vars[2][1:]
    var_dict["E"] = vars[3][1:]
    var_dict["LR"] = vars[4][2:]
    var_dict["LRD"] = vars[5][3:]
    var_dict["S"] = vars[6][1:-4]

    return var_dict


def load_data(maindir,year):

    # trajectories
    traj_paths = os.path.join(maindir, 'data', f'V{year}')
    trajectories = [np.load(p, allow_pickle=True) for p in
                    np.sort(glob.glob(os.path.join(traj_paths, '**', 'traj.npy')))]

    # feature matrices
    fmat_paths = os.path.join(maindir, 'data', f'V{year}')
    feature_matrices = [np.load(p, allow_pickle=True) for p in
                        np.sort(glob.glob(os.path.join(fmat_paths, '**', 'fmat.npy')))]

    # transition prob: they are all the same so just pick one
    Tprob = np.load(os.path.join(maindir, 'data', f'transition_prob_Y{year}.npy'))

    # TODO: temp
    Tprob = Tprob[:-2, :, :-2]
    Tprob = np.ascontiguousarray(Tprob)

    return feature_matrices, Tprob, trajectories


def read_multi_data(maindir,year,kind="theta"):
    files = glob.glob(os.path.join(maindir,f"{kind}*V{year}*N*E*LR*LRD*S*.npy"))
    data_list = []
    for f in files:
        #var_dict = {k:v for k,v in [split_fname(s) for s in f.split("_")]}
        #var_dict['fname'] = f
        var_dict = split_fname2(f)
        data_list.append(var_dict)
    return data_list


def average_multi_seed_estimation(maindir,year,**run_params):

    # Must exclude param that is to be averaged over (e.g. seed)

    df = pd.DataFrame(read_multi_data(maindir, year))
    df = df.loc[(df[list(run_params)] == pd.Series(run_params)).all(axis=1)]
    fnames = df.fname.values
    weights = np.zeros(df.shape)
    for i,f in enumerate(fnames):
        w_array = np.load(os.path.basename(f)) # last updated weights
        weights[i,:] = w_array[-1].mean(axis=(0,1))

    return weights.mean(axis=0)


def process_weights(path,label):

    weights = np.load(path)

    N = weights.shape[-1]
    colnames = [f"F{i+1}" for i in range(N)]

    weights = weights[-1].mean(axis=(0, 1)).reshape((1,N))

    wdf = pd.DataFrame(weights,columns=colnames)
    wdf['Player'] = label
    wdf_long = pd.melt(wdf, id_vars="Player", var_name="Features", value_name="Weight")

    return wdf_long


def print_run_params(**params):
    print("** PARAMETERS ** ")
    for k,v in params.items():
        print(f"{k}: {v}")
    print("**            ** ")