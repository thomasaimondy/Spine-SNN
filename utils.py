import numpy as np
import torch
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)): # 最多是100w 多的数据会把之前的数据覆盖
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)



def make_paraset(nps):

    params_A = [[0.24261412, 0.103609,0.00336895,0.2556423],         #固定初始化 k=1  seed=1             A0
                [-0.00088873, 0.12460965, -0.3259607, 0.08871528],   #均匀随机初始化(-0.5~0.5) k=1 seed=1   A1
                [0.35401717, 0.16196088, -0.13909505, 0.19326942],   #均匀随机初始化(-0.5~0.5) k=1 seed=3   A2
                [0.18927404, 0.03311339, 0.24647327, 0.2535481],     # 均匀随机初始化(-0.5~0.5) k=1 seed=4  A3
                [0.07122789, 0.00882253, -0.03069546, 0.2122931],    # 均匀随机初始化(-0.5~0.5) k=1 seed=5  A4
                ]



    if nps == 'A0':
        paraset = [params_A[0]]
    elif nps == 'A1':
        paraset = [params_A[1]]
    elif nps == 'A2':
        paraset = [params_A[2]]
    elif nps == 'A3':
        paraset = [params_A[3]]
    elif nps == 'A4':
        paraset = [params_A[4]]



    return paraset