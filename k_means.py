import matplotlib.pyplot as plt
import torch
import os
from sklearn.cluster import KMeans
from sklearn import metrics

def preprocess(seed,flag):
    path = './params/td3-spikeAdeepC-Ant-v3--spike_ts-10-none-encoder-dim-10-Pretrain'+flag
    filename = 'pretrain/model' + str(seed) + '_paraset.pth'

    paraset = torch.load(os.path.join(path, filename))

    A = []
    B = []
    C = []
    D = []
    for i in range(3):
        a = paraset['a' + str(i)]
        b = paraset['b' + str(i)]
        c = paraset['c' + str(i)]
        d = paraset['d' + str(i)]
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)

    A = torch.cat(A, dim=1)    # (1, 256*2+act_dim*10)
    B = torch.cat(B, dim=1)
    C = torch.cat(C, dim=1)
    D = torch.cat(D, dim=1)


    X1 = torch.cat([A, B], dim=0).detach().cpu().numpy()
    X1 = X1.T

    X2 = torch.cat([C, D], dim=0).detach().cpu().numpy()
    X2 = X2.T

    return X1,X2



def visual(X):
    plt.scatter(X[:,0],X[:,1],marker='o')
    plt.show()

def kmeans(X,note):


    plt.rcParams['font.size'] = 14


    k_means = KMeans(n_clusters=1,random_state=10)
    k_means.fit(X)
    cluster_center = k_means.cluster_centers_


    print("n_cluster==1")
    print("cluster_centers: ", cluster_center)
    plt.scatter(X[:, 0], X[:, 1], c='b', marker='o',alpha=0.6)
    plt.scatter(cluster_center[:, 0], cluster_center[:, 1], marker='^', c='r',alpha=1)
    if note=='1':
        plt.title(r"Candidate $(\theta_a,\theta_b)$ from Ant-v3")
        plt.xlabel(r"The range of $\theta_a$")
        plt.ylabel(r"The range of $\theta_b$")
        #plt.savefig(fname="./fig_pretrain/fig2.svg", format="svg")
        plt.show()
    else:
        plt.title(r"Candidate $(\theta_c,\theta_d)$ from Ant-v3")
        plt.xlabel(r"The range of $\theta_c$")
        plt.ylabel(r"The range of $\theta_d$")
        #plt.savefig(fname="./fig_pretrain/fig3.svg", format="svg")
        plt.show()




if __name__=='__main__':
    seed = 5
    flag = '2'    # 三种初始化方式      ''  '1'  '2'
    #当聚类中心=1时  相当于对所有列求均值， 此时4个一起聚类 还是前两个聚一个，后两个聚一个 结果是一样的


    X1,X2 = preprocess(seed,flag)
    #visual(X1)
    kmeans(X1, note='1')

    print("**" * 40)

    #visual(X2)
    kmeans(X2, note='2')

















