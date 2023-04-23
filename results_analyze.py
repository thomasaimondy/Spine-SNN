import matplotlib.pyplot as plt
import pickle
import numpy as np


def smooth_training_data(training_data, smooth_half_window):
    """
    Smooth training data using average of smooth window
    :param training_data: list of training data
    :param smooth_half_window: half window size
    :return: smooth_data
    """
    data_num = training_data.shape[0]
    smooth_data = np.zeros(data_num)
    for num in range(data_num):
        if num < smooth_half_window:
            smooth_data[num] = np.sum(training_data[:num+smooth_half_window+1]) / (num + smooth_half_window + 1)
        elif num >= data_num - smooth_half_window:
            smooth_data[num] = np.sum(training_data[num-smooth_half_window:]) / (data_num - num + smooth_half_window)
        else:
            smooth_data[num] = np.sum(training_data[num-smooth_half_window:num+smooth_half_window+1]) / (2 * smooth_half_window + 1)
    return smooth_data


def read_multiple_ppo_mujoco_models(model_dir, model_num, steps, smooth_factor=5):
    test_reward_list = np.zeros((steps, model_num))


    for num in range(model_num):
        if model_num>1:
            index = num
        elif model_num==1: # Ant-v3 学习MDN 使用的是seed1
            index = num+1

        test_reward, _ = pickle.load(open(model_dir + '/model' + str(index) + '_test_rewards.p', 'rb'))
        smooth_test_reward = smooth_training_data(np.array(test_reward), smooth_factor)
        test_reward_list[:, num] = smooth_test_reward

    return test_reward_list.mean(axis=1), test_reward_list.std(axis=1)


def plot_multiple_mean_rewards(model_dir_list, label_list, color_list, model_num, steps, env_name,scale=2):
    plt.rcParams['font.size'] = 14
    plt.figure()
    for i, model_dir in enumerate(model_dir_list, 0):
        label = label_list[i]
        color = color_list[i]
        reward_mean, reward_std = read_multiple_ppo_mujoco_models(model_dir, model_num, steps,smooth_factor=5)  #绘图时需要平滑
        reward_mean1, reward_std1 = read_multiple_ppo_mujoco_models(model_dir, model_num, steps,smooth_factor=0) #计算max average reward时不需要平滑
        plt.plot([num for num in range(steps)], reward_mean, color, label=label)
        plt.fill_between([num for num in range(steps)], reward_mean - reward_std/scale, reward_mean + reward_std/scale,
                         alpha=0.2, color=color)
        print("smooth value: ",np.max(reward_mean), reward_std[np.argmax(reward_mean)])
        print("non-smooth value: ",np.max(reward_mean1),reward_std1[np.argmax(reward_mean1)])  # Max average return = 不平滑的情况下 曲线上的最高点对应的值
    plt.xlim([0, steps])


    if model_num >1 :

        plt.xlabel("Training steps (x10k)")
        plt.ylabel("Average rewards")
        plt.title(env_name)
        plt.legend(loc='lower right')
        plt.savefig(fname="./neuron/fig_new4.svg", format="svg")
    else:
        plt.xlabel("Training steps (x10k)")
        plt.ylabel("Rewards")
        plt.title("Training curve in Ant-v3")

        plt.savefig(fname="./fig_pretrain/fig1.svg", format="svg")
    plt.show()

if __name__ == "__main__":

    env_name = 'Hopper-v3' #  Ant-v3 、HalfCheetah-v3 、 Walker2d-v3、Hopper-v3
    model_num = 10 #10个随机种子/模型  0-9
    steps = 100 #100w步 每1w步一个值
    experience = 'main'

    if experience == 'main':

        if env_name=='Ant-v3' or env_name=="HalfCheetah-v3": # PopSAN 前两个环境使用 pop+regular(det) 后两个使用的是 pop+poisson
            coding = 'regular'
        else:
            coding = 'poisson'
        '''
        model_dir_list = ['./params/td3-deepAC-'+env_name, './params/td3-deepAC-'+env_name+'-pop', './params/td3-deepAC-'+env_name+'-pop1','./params/td3-spikeAdeepC-'+env_name+'-population-spike_ts-5-'+coding+'-encoder-dim-10-decoder-dim-10',
                                './params/td3-spikeAdeepC-'+env_name+'-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10-A1',
                                ]
        labels_list = ['TD3','TD3-pop','TD3-pop1','TD3-PopSAN','TD3-PMSAN']
        color_list = ['red','black','blue','green','purple']
        '''
        model_dir_list = ['./params/td3-deepAC-' + env_name,
                          './params/td3-deepAC-' + env_name + '-pop1',
                          './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-' + coding + '-encoder-dim-10-decoder-dim-10',
                          './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10-A1',
                          ]
        labels_list = ['TD3', 'TD3-Pop','TD3-PopSAN','TD3-PDSAN']
        color_list = ['red', 'blue','green', 'purple']

    elif experience=='neuron': #固定编码方式为pop-ori  比较MDN(A1)和LIF

        model_dir_list = ['./params/td3-spikeAdeepC-'+env_name+'-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10' #LIF
            ,'./params/td3-spikeAdeepC-'+env_name+'-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10-A1'] #A1
        if env_name == 'Hopper-v3':
            model_dir_list[0] += '-LIF'
        labels_list = ['LIF','DNs']
        color_list = ['red','purple']

    elif experience=='coding1':  #固定神经元为 DN(A1) 比较各种编码方式
        model_dir_list = [
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-poisson-encoder-dim-1-decoder-dim-10-A1',
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-regular-encoder-dim-10-decoder-dim-10-A1',
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-uniform-encoder-dim-10-decoder-dim-10-A1',
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-poisson-encoder-dim-10-decoder-dim-10-A1',
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10-A1',
        ]

        labels_list = ['poi','pop-det', 'pop-uni', 'pop-poi', 'pop']
        color_list = ['orange','red','blue','green','purple']

    elif experience=='coding2': #固定编码方式为pop  神经元为A1 比较pop的大小 2，5，10
        assert env_name == 'Ant-v3'

        model_dir_list = [
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-none-encoder-dim-2-decoder-dim-10-A1',
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-none-encoder-dim-5-decoder-dim-10-A1',
            './params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10-A1',
        ]
        labels_list = ['p=2','p=5','p=10']
        color_list = ['red','green','purple']

    elif experience=='pretrain':
        model_num = 1
        assert  env_name=='Ant-v3'
        model_dir_list = ['./params/td3-spikeAdeepC-' + env_name + '-population-spike_ts-5-none-encoder-dim-10-decoder-dim-10-Pretrain']
        labels_list = ['Ant-v3_MDN_Learning']
        color_list = ['blue']

    plot_multiple_mean_rewards(model_dir_list, labels_list, color_list, model_num, steps, env_name)





