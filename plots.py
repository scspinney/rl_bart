import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import os


def plot_weights(weights):

    df = pd.DataFrame({'weights': weights})
    df['positive'] = df['weights'] > 0
    df['features'] = range(1, 13)
    df.set_index("features", drop=True, inplace=True)

    df['weights'].plot(x='features',
                       kind='bar',
                       color=df.positive.map({True: 'g', False: 'r'}))

    plt.savefig(f'results/weights{str(datetime.date.today())}.png')
    plt.show()


def plot_reward_landscape(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,weights,feature_matrices,obs_exp_rewards,avg_save_state,kind='line'):


    sns.set(style="darkgrid")
    out_name = f'results/reward_landscape-{str(datetime.date.today())}.npy'

    if not os.path.exists(out_name):

        contrast = np.ones((N_FEAT-1,))
        #contrast[10] = -1
        rewards=np.zeros((N_TRIAL,N_EXPERTS,N_STATES))

        # retrieve reward landscape
        for t in range(N_TRIAL):
            for e in range(N_EXPERTS):
                fmat = feature_matrices[e][t][:-2,:-1]
                rewards[t,e,:] = np.dot(np.multiply(contrast,fmat), weights)

        np.save(out_name, rewards, allow_pickle=True)
    else:
        rewards = np.load(out_name)

    x = np.arange(1,129)

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

        plt.figure(dpi=1200)
        plt.figure(figsize=(20, 10))
        ax = sns.relplot(x="state",
                  y="exp_pred_reward",
                  hue="balloon",
                  kind="line",
                  data=rewards,
                  facet_kws={"legend_out": False})

        plt.plot(x,obs_exp_rewards,color='g',label="Observed Average Reward")
        plt.plot(x,expected_rewards,color='pink',label="Expected reward")


        # add red line at positive threshold
        ax.ax.axvline(pos_thres, color='red')
        ax.ax.axvline(avg_save_state, color='blue')

        # add xtick at that pos thresh
        #x_ticks = np.append(ax.ax.get_xticks(), (pos_thres,avg_save_state))
        #x_ticks = x_ticks[1:]

        plt.annotate(f"{int(pos_thres)}",xy=(pos_thres-7,-30),color='r',size=8)
        plt.annotate(f"{int(avg_save_state)}", xy=(avg_save_state + 2, -30), color='b', size=8)

        # Set xtick locations to the values of the array `x_ticks`
        #ax.ax.set_xticks(x_ticks)
        plt.xticks(fontsize=7,rotation=45)

        plt.title(f"Reward distribution over states, averaged across the {N_EXPERTS} experts.")
        txt = "Red vline indicates largest pump number (state) with positive reward. Blue vline is the average save state"
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=7)

        plt.xlabel("States")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(f'results/reward_landscape{str(datetime.date.today())}.png')

        plt.show()





def plot_gradients(gradients):


    #TODO: hard coded fix but rerun with proper gradient sizing
    gradients = gradients[:,:,:30,:]

    grad_epoch1 = gradients[0]
    grad_epoch1 = grad_epoch1.reshape(-1, grad_epoch1.shape[-1])
    grad_epoch2 = gradients[1]
    grad_epoch2 = grad_epoch2.reshape(-1, grad_epoch2.shape[-1])
    grad_epochs = np.concatenate((grad_epoch1,grad_epoch2),axis=0)
    gdf = pd.DataFrame(grad_epochs)

    #gdf = pd.DataFrame(np.concatenate((grad_epoch1,grad_epoch2)).reshape(-1,gradients.shape[-1]))
    gdf['epoch'] = pd.Series(np.repeat(np.array([1,2]), [grad_epoch1.shape[0],grad_epoch2.shape[0]]))
    #gdf = pd.DataFrame(gradients.reshape(-1, gradients.shape[-1]))
    gdf.reset_index(inplace=True)
    gdf.rename({'index':'iterations'},axis=1,inplace=True)


    # Initialize the figure
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    plt.figure(dpi=1200)
    plt.figure(figsize=(40,40))
    # general title
    plt.suptitle("Gradients of feature weights", fontsize=13, color='black', y=1.02)

    num = 0
    for column in gdf.drop(['iterations','epoch'], axis=1):
        num += 1

        # Find the right spot on the plot
        plt.subplot(3, 4, num)

        # Plot the lineplot
        sns.lineplot(data=gdf, x='iterations',y=gdf[column], hue='epoch')
        #plt.plot(gdf['iterations'], gdf[column], marker='', color=palette(num), linewidth=1.4, alpha=0.7, label=column)

        #increase tick size



        # Not ticks everywhere
        if num in range(9):
            plt.tick_params(labelbottom='off')
        if num not in [1, 5, 9]:
            plt.tick_params(labelleft='off')
        plt.tick_params(labelsize='large')

        # Add title
        plt.title(f'Feature: {column}', loc='center', fontsize=24, fontweight=1)



    plt.tight_layout()
    plt.savefig(f'results/gradients{str(datetime.date.today())}.svg')
    plt.show()
