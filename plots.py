import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from numba import njit
import os


def plot_ll(lldf):
    g = sns.FacetGrid(lldf, col="LRD", row="S",sharex=True,sharey=False)
    g.map(sns.barplot,"LR","LL")

    #sns.FacetGrid(row="S",col="LRD",data=lldf)
    #sns.barplot(x="LR",y="LL",data=lldf)
    plt.tight_layout()
    plt.savefig(f'results/LL_{str(datetime.date.today())}.png')
    plt.show()



def plot_multi_weights(data_list,outname=None):


    if outname is None:
        outname = f"results/multi_run_data{str(datetime.date.today())}.csv"


    if not os.path.exists(outname):

        wdict = {'year':[],
                  'lr':[],
                  'lr_decay':[],
                 'epoch':[],
                 'expert':[],
                 'feature':[],
                 'weight':[],
                 'seed':[]}

        for run in data_list:
            print(run)
            weights = np.load(run['fname'])

            N_EPOCHS, N_EXPERTS, N_TRIALS, N_FEAT = np.shape(weights)

            # take every 20th epoch
            for epoch in range(0, N_EPOCHS, 20):
                for f in range(N_FEAT):
                    for e in range(N_EXPERTS):
                        for b in range(N_TRIALS):

                            wdict['year'].append(run['V'])
                            wdict['lr'].append(run['LR'])
                            wdict['lr_decay'].append(run['LRD'])
                            wdict['epoch'].append(epoch)
                            wdict['expert'].append(e)
                            wdict['feature'].append(f)
                            wdict['weight'].append(weights[epoch, e, b, f])
                            wdict['seed'].append(run['S'])

        wdf = pd.DataFrame().from_dict(wdict)
        wdf.to_csv(outname)
    else:
        wdf = pd.read_csv(outname)

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(80, 40), dpi=1200)

    sns.relplot(x="epoch", y="weight",
                hue="lr_decay", col="feature",
                row="seed",style="lr",
                #height=5, aspect=.75,
                facet_kws=dict(sharey=False),
                kind="line", legend="brief", data=wdf)

    plt.tight_layout()
    plt.savefig(f'results/multi_weights{str(datetime.date.today())}.png')
    plt.show()




def plot_weights(weights,kind='line'):


    out_d1 = f'results/weightdf1{str(datetime.date.today())}.csv'
    out_d2 = f'results/weightdf2{str(datetime.date.today())}.csv'
    out_d3 = f'results/weightdf3{str(datetime.date.today())}.csv'

    N_EPOCHS, N_EXPERTS, N_TRIALS, N_FEAT = np.shape(weights)

    avg_weights_e = weights.mean(axis=1)
    avg_weights_b = weights.mean(axis=2)

    wdict1 = {'epoch':[],
             'balloon':[],
             'expert':[],
             'feature':[],
             'weight':[]}

    wdict2 = {'epoch':[],
             'balloon':[],
             'feature':[],
             'weight':[]}

    wdict3 = {'epoch':[],
             'expert':[],
             'feature':[],
             'weight':[]}

    if any([not os.path.exists(p) for p in [out_d1,out_d2,out_d2]]):

        for epoch in range(0,N_EPOCHS,20):
            for f in range(N_FEAT):
                for e in range(N_EXPERTS):

                    wdict3['epoch'].append(epoch)
                    wdict3['expert'].append(e)
                    wdict3['feature'].append(f)
                    wdict3['weight'].append(avg_weights_b[epoch, e, f])
                    # for b in range(N_TRIALS):
                    #
                    #     wdict1['epoch'].append(epoch)
                    #     wdict1['balloon'].append(b)
                    #     wdict1['expert'].append(e)
                    #     wdict1['feature'].append(f)
                    #     wdict1['weight'].append(weights[epoch,e,b,f])
                    #
                    #     wdict2['epoch'].append(epoch)
                    #     wdict2['balloon'].append(b)
                    #     wdict2['feature'].append(f)
                    #     wdict2['weight'].append(avg_weights_e[epoch, b, f])


        #wdf1 = pd.DataFrame().from_dict(wdict1)
        #wdf2 = pd.DataFrame().from_dict(wdict2)
        wdf3 = pd.DataFrame().from_dict(wdict3)

        # save
        wdf3.to_csv(out_d3)

    else:
        #wdf1 = pd.read_csv(out_d1)
        #wdf2 = pd.read_csv(out_d2)
        wdf3 = pd.read_csv(out_d3)

        # Initialize the figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(80, 40), dpi=1200)

    if kind == 'bar':
        pass
        # sns.catplot(x="feature", y="weight",
        #             hue="expert", col="epoch",
        #             data=wdf1, kind="bar",
        #             legend_out= False,
        #             legend=False)

    elif kind == 'line':
        # g = sns.FacetGrid(data=wdf, col="feature", hue="balloon")
        # g.map(sns.pointplot, "epoch", "weight")

        # Plot the lines on two facets
        # sns.relplot(x="epoch", y="weight",
        #             hue="balloon", col="feature",
        #             #height=5, aspect=.75,
        #             facet_kws=dict(sharey=False),
        #             kind="line", legend="brief", legend_out=True,data=wdf2)

        sns.relplot(x="epoch", y="weight",
                    hue="expert", col="feature",
                    #height=5, aspect=.75,
                    facet_kws=dict(sharey=False),
                    kind="line", legend="brief", data=wdf3)

    plt.tight_layout()
    plt.savefig(f'results/weights{str(datetime.date.today())}.png')
    plt.show()




def plot_reward_landscape(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,weights,feature_matrices,obs_exp_rewards,avg_save_state,kind='line',clobber=True):


    sns.set(style="darkgrid")
    out_name = f'results/reward_landscape-{str(datetime.date.today())}.npy'

    if not os.path.exists(out_name) or clobber :

        contrast = np.ones((N_FEAT,))
        rewards=np.zeros((N_TRIAL,N_EXPERTS,N_STATES))

        # retrieve reward landscape
        for t in range(N_TRIAL):
            for e in range(N_EXPERTS):
                fmat = feature_matrices[e][t][:-2]
                rewards[t,e,:] = np.dot(np.multiply(contrast,fmat), weights)

        np.save(out_name, rewards, allow_pickle=True)
    else:
        rewards = np.load(out_name)

    x = np.arange(1,129)

    avg_pred_reward = rewards.mean(axis=1).mean(axis=0)
    rewards = pd.DataFrame(rewards.mean(axis=1))

    # add balloon number identifier
    rewards['balloon'] = pd.Series([b % 30 for b in range(len(rewards))])

    # wide to long
    rewards = rewards.melt(id_vars=['balloon'],var_name="state",value_name="reward")
    rewards['exp_pred_reward'] = rewards.apply(lambda x: x.reward * (128 - x.state + 1) / (129 - x.state + 1), axis=1)
    expected_rewards = [(1290-20*i)/(129-i) for i in range(1,129)]
    expected_rewards[100:] = [np.nan for _ in range(100,128)]

    # on the last balloon (30), what is the largest state to still be positive in reward
    # and plot that line in red
    pos_thres = max(rewards.state[(rewards.reward > 0) & (rewards.balloon == 29)])
    if kind == 'line':

        plt.figure(figsize=(60, 40),dpi=900)

        ax = sns.relplot(x="state",
                  y="exp_pred_reward",
                  #y="reward",
                  hue="balloon",
                  kind="line",
                  data=rewards,
                  facet_kws={"legend_out": False})

        ax.ax.plot(x,obs_exp_rewards,color='green',label="Observed average reward")
        ax.ax.plot(x, avg_pred_reward, color='blue', label="Average predicted reward")
        ax.ax.plot(x,expected_rewards,color='pink',label="Expected reward")
        #plt.plot(x,obs_exp_rewards,style='--',label="Observed Average Reward")
        #plt.plot(x,expected_rewards,color='pink',label="Expected reward")


        # add red line at positive threshold
        ax.ax.axvline(pos_thres, color='red', alpha=0.2)
        ax.ax.axvline(avg_save_state, color='blue', alpha=0.2)

        # add xtick at that pos thresh
        #x_ticks = np.append(ax.ax.get_xticks(), (pos_thres,avg_save_state))
        #x_ticks = x_ticks[1:]

        ax.ax.annotate(f"{int(pos_thres)}",xy=(pos_thres+3,-20),color='r',size=8)
        ax.ax.annotate(f"{int(avg_save_state)}", xy=(avg_save_state - 8, -20), color='b', size=8)

        # Set xtick locations to the values of the array `x_ticks`
        #ax.ax.set_xticks(x_ticks)
        plt.xticks(fontsize=7,rotation=45)

        plt.suptitle(f"Reward distribution over states, averaged across the {N_EXPERTS} experts.",fontsize=8)
        txt = "Red vline indicates largest pump number (state) with positive reward. Blue vline is the average save state"
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=7)

        plt.xlabel("States")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize = 'xx-small')
        plt.savefig(f'results/reward_landscape{str(datetime.date.today())}.png')
        plt.show()


def plot_gradients(gradients):


    #TODO: hard coded fix but rerun with proper gradient sizing
    avg_grads = gradients.mean(axis=1)

    N_EPOCHS, N_TRIALS, N_ITER, N_FEAT = np.shape(avg_grads)

    gdict = {'epoch':[],
             'balloon':[],
             'iteration':[],
             'feature':[],
             'gradient':[]}

    for epoch in range(N_EPOCHS):
        for b in range(N_TRIALS):
            for iter in range(N_ITER):
                for f in range(N_FEAT):
                    gdict['epoch'].append(epoch)
                    gdict['balloon'].append(b)
                    gdict['iteration'].append(iter)
                    gdict['feature'].append(f)
                    gdict['gradient'].append(avg_grads[epoch,b,iter,f])

    gdf = pd.DataFrame().from_dict(gdict)

    # Initialize the figure
    plt.style.use('seaborn-darkgrid')

    fig, axs = plt.subplots(N_EPOCHS+1, N_FEAT, figsize=(100,40), sharex=True, sharey=False)

    colors = plt.cm.coolwarm(np.linspace(0, 1, N_TRIALS))
    line_labels = [i+1 for i in range(N_TRIALS)]
    for e in range(N_EPOCHS):
        for f in range(N_FEAT):
            num = N_FEAT*e+f + 1
            f_gdf = gdf[(gdf['feature'] == f) & (gdf['epoch'] == e)]

            for i, (name, group) in enumerate(f_gdf.groupby("balloon")):
                axs[e,f].plot(group['iteration'], group['gradient'], color=colors[i], label=f"Balloon {i}")
                #axs[f].plot(group['iteration'], group['gradient'], color=colors[i], label=f"Balloon {i}")

            axs[e,f].set_title(f"Epoch: {e}, Feature: {f}")
            #axs[f].set_title(f"Epoch: {e}, Feature: {f}")

            # Not ticks everywhere
            if num in range(N_FEAT):
                axs[e,f].tick_params(labelbottom='off')
                #axs[f].tick_params(labelbottom='off')
            if num not in [1, N_FEAT+1]:
                axs[e,f].tick_params(labelleft='off')
                #axs[f].tick_params(labelleft='off')
            axs[e,f].tick_params(labelsize='small')
            #axs[f].tick_params(labelsize='small')

            if e == 0 and f == 10:
                axs[e,f].legend()
                #axs[f].legend()



    fig.tight_layout()
    plt.savefig(f'results/gradients{str(datetime.date.today())}.png')
    plt.show()
