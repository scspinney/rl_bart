import numpy as np
import pandas as pd


def evaluate_recovery(actual_path,recovered_path):

    #actual = np.load(actual_path)
    #actual = actual[-1,-1,-1,:]
    actual = [0.49049365,  0.97522182,  0.81013244,  0.59797393,  0.16992047,
              0.15576953,  0.06382926,  0.87656816,  0.60628161,  0.71614336,
              -0.044085]

    recovered = np.load(recovered_path)
    recovered  = recovered [-1, -1, -1, :]

    features = pd.Series(["1: # of times being in this state",
                          "2: whether this state was burst in previous trial",
                          "3: whether this state was save in previous trial",
                          "4: whether this state was burst in 2nd previous trial",
                          "5: whether this state was save in 2nd previous trial",
                          "6: whether this state was burst in 3rd previous trial",
                          "7: whether this state was save in 3rd previous trial",
                          "8: whether is average burst status",
                          "9: whether is average save status",
                          "10: whether is average end status",
                          "11: # of steps (pumps) in current trial"])
    df = pd.DataFrame({'Feature':features,
                       'Actual weights': actual,
                       'Recovered weights': recovered})

    df['Error'] = df['Actual weights'] - df['Recovered weights']
    df['Percentage error'] = 100*(df['Error']/df['Actual weights'])

    df.to_html("results/evaluate_recovery.htmll")


actual_path = ''
recovered_path = 'results/theta_V2_N138_E500_LR0.0001_LRD1_S42.npy'
