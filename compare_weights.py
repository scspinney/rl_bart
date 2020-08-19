import numpy as np
import pandas as pd
from plots import plot_weights_bar


def process_weights(path,label):

    weights = np.load(path)

    N = weights.shape[-1]
    colnames = [f"F{i+1}" for i in range(N)]

    weights = weights[-1].mean(axis=(0, 1)).reshape((1,N))

    wdf = pd.DataFrame(weights,columns=colnames)
    wdf['Player'] = label
    wdf_long = pd.melt(wdf,id_vars=['Player'],var_name="Features",value_name="Weight")

    return wdf_long

human_weights = process_weights('results/theta_V2_N138_E10_LR0.0001_LRD1_S42.npy','humans')
ql_weights = process_weights('results/ai_theta_V2_N138_E10_LR0.0001_LRD1_S42.npy','qlearner')

all_weights = human_weights.append(ql_weights)
plot_weights_bar(all_weights)


