import random

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SpineEpspEncoderNoneEpsp1(nn.Module):
    """ Learnable Population Coding Epsp Encoder with Regular Epsp Trains """
    def __init__(self, obs_dim, spine_dim, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param spine_dim: population dimension
        :param Epsp_ts: Epsp timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.spine_dim = spine_dim
        self.encoder_neuron_num = obs_dim * spine_dim
        self.device = device
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, spine_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (spine_dim - 1)
        for num in range(spine_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, spine_dim) + std
        self.mean_encoder = nn.Parameter(tmp_mean)
        self.std_encoder = nn.Parameter(tmp_std)

    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: spine_Epsps
        """
        obs = obs.view(-1, self.obs_dim, 1)  # (batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        spine_act=1/(1+torch.exp(-(obs-self.mean_encoder)/self.std_encoder)).view(-1, self.encoder_neuron_num)
        # Generate Poisson Epsp Trains
        return spine_act

class EpspDecoder(nn.Module):
    """ Population Coding Epsp Decoder """
    def __init__(self, act_dim, spine_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param spine_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.spine_dim = spine_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, spine_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, spine_act):
        """
        :param spine_act: output population activity
        :return: raw_act
        """
        spine_act = spine_act.view(-1, self.act_dim, self.spine_dim)
        raw_act = self.output_activation(self.decoder(spine_act).view(-1, self.act_dim))
        return raw_act

class ActorSpine(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256], encoder_spine_dim=10,
                 decoder_dim=10, mean_range = (-3, 3), std = math.sqrt(0.15), device = torch.device('cpu'), actorLR1=1e-5, actorLR2=1e-5):
        super(ActorSpine, self).__init__()
        print("DeepActor+Spine")
        self.encoder = SpineEpspEncoderNoneEpsp1(state_dim, spine_dim=encoder_spine_dim, mean_range=mean_range, std=std,
                                                     device=device)
        self.l1 = nn.Linear(state_dim*encoder_spine_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim * decoder_dim)

        self.decoder = EpspDecoder(action_dim, decoder_dim)
        self.max_action = max_action

    def forward(self, state):
        in_spine = self.encoder(state)
        a = F.relu(self.l1(in_spine))
        a = F.relu(self.l2(a))
        a = self.l3(a)  #不加激活函数
        b=self.decoder(a)
        for i in range(0,b.shape[0]):
            for j in range(0,b.shape[1]):
                if torch.isnan(b[i][j]):
                    b[i][j]=random.random()

        return self.max_action * self.decoder(a)  # 有内置的tanh