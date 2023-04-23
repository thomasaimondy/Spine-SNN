import torch.nn as nn
import torch
import torch.nn.functional as F
import math



class PopSpikeEncoderNoneSpike1(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.device = device
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  #(batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num) #(batch_size,obs_dim*pop_dim)

        return pop_act

class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act

class ActorPop(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256],encoder_pop_dim=10,
                 decoder_pop_dim=10, mean_range = (-3, 3), std = math.sqrt(0.15), device = torch.device('cpu')):
        super(ActorPop, self).__init__()
        print("DeepActor+Pop")
        self.encoder = PopSpikeEncoderNoneSpike1(state_dim, pop_dim=encoder_pop_dim, mean_range=mean_range, std=std,
                                                     device=device)
        self.l1 = nn.Linear(state_dim*encoder_pop_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim*decoder_pop_dim)

        self.decoder = PopSpikeDecoder(action_dim, decoder_pop_dim)
        self.max_action = max_action

    def forward(self, state):
        in_pop = self.encoder(state)
        a = F.relu(self.l1(in_pop))
        a = F.relu(self.l2(a))
        a = self.l3(a)  #不加激活函数

        return self.max_action * self.decoder(a)  # 有内置的tanh