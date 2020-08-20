import numpy as np
import pandas as pd
from plots import plot_weights_bar
from utils import *

def process_weights(path,label):

    weights = np.load(path)

    N = weights.shape[-1]
    colnames = [f"F{i+1}" for i in range(N)]

    weights = weights[-1].mean(axis=(0, 1)).reshape((1,N))

    wdf = pd.DataFrame(weights,columns=colnames)
    wdf['Player'] = label
    wdf_long = pd.melt(wdf,id_vars=['Player'],var_name="Features",value_name="Weight")

    return wdf_long

human_weights = process_weights('results/theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','humans')

ai_weights1 = process_weights('results/QL_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','qlearner')
ai_weights2 = process_weights('results/AlwaysPump_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','alwayspump')
ai_weights3 = process_weights('results/Random_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy','random')

ai_weights = ai_weights1.append(ai_weights2.append(ai_weights3))
#ai_weights = ai_weights1.append(ai_weights3)

all_weights = human_weights.append(ai_weights)
plot_weights_bar(all_weights)


