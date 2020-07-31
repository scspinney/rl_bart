import numpy as np
import glob as glob
import os



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