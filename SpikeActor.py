
import math

import torch
import torch.nn as nn
from utils import make_paraset
"""
Parameters for SNN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

#群编码有参数 所以需要写伪BP
class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PseudoEncoderSpikePoisson(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Poisson Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PseudoEncoderSpikeUniform(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Poisson Spike for encoder """
    @staticmethod
    def forward(ctx, input):  #input (batch_size,obs_dim*pop_dim)
        return (input > torch.rand(size=input.shape,device=input.device)).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PopSpikeEncoderNoneSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
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
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        if pop_dim>1:
            delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
            for num in range(pop_dim):
                tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        elif pop_dim==1:
            tmp_mean[0, :, 0] = 0
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  # (batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1,
                                                                                          self.encoder_neuron_num)  # (batch_size,obs_dim*pop_dim)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts,
                                 device=self.device)  # (batch_size,obs_dim*pop_dim,spike_ts)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            # pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
            pop_spikes[:, :, step] = pop_act  # 其实没必要写PseudoEncoderSpikeNone，没有转换脉冲序列 过程是可导的，可以直接使用BP
        return pop_spikes

class PopSpikeEncoderPoissonSpike(PopSpikeEncoderNoneSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikePoisson.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes

class PopSpikeEncoderUniformSpike(PopSpikeEncoderNoneSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikeUniform.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  #(batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num) #(batch_size,obs_dim*pop_dim)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device) #(batch_size,obs_dim*pop_dim,spike_ts)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes

class PopSpikeEncoderRegularSpike(PopSpikeEncoderNoneSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1,
                                                                                          self.encoder_neuron_num)  # 均值和标准差是参数 为了让高斯分布输出在0-1（e^(-x),x>=0）之间，前面没有加系数
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes

class SpineSpikeEncoderNoneSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
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
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        if pop_dim>1:
            delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
            for num in range(pop_dim):
                tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        elif pop_dim==1:
            tmp_mean[0, :, 0] = 0
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  # (batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act=1/(1+torch.exp(-(obs-self.mean))).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts,
                                 device=self.device)  # (batch_size,obs_dim*pop_dim,spike_ts)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            # pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
            pop_spikes[:, :, step] = pop_act
        return pop_spikes

class SpineSpikeEncoderPoissonSpike(SpineSpikeEncoderNoneSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikePoisson.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act=1/(1+torch.exp(-(obs-self.mean))).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes

class SpineSpikeEncoderUniformSpike(SpineSpikeEncoderNoneSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikeUniform.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  #(batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act=1/(1+torch.exp(-(obs-self.mean))).view(-1, self.encoder_neuron_num) #(batch_size,obs_dim*pop_dim)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device) #(batch_size,obs_dim*pop_dim,spike_ts)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes

class SpineSpikeEncoderRegularSpike(SpineSpikeEncoderNoneSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  #(batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num) #(batch_size,obs_dim*pop_dim)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device) #(batch_size,obs_dim*pop_dim,spike_ts)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            #pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
            pop_spikes[:, :, step] = pop_act      #其实没必要写PseudoEncoderSpikeNone，没有转换脉冲序列 过程是可导的，可以直接使用BP
        return pop_spikes

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


class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device,ntype):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.ntype = ntype

        self.pseudo_spike = PseudoSpikeRect.apply
        print(ntype)

        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.ntype != 'LIF':
            if self.ntype=='Pretrain':
                self.paraset = nn.ParameterDict({})   #模型参数
                a, b, c, d = self.generate_parasets(hidden_sizes[0])
                self.paraset.update({'a0': a})
                self.paraset.update({'b0': b})
                self.paraset.update({'c0': c})
                self.paraset.update({'d0': d})
            else:
                self.paraset = {}  # 不是模型参数 （固定）
                a, b, c, d = self.generate_parasets(hidden_sizes[0])
                self.paraset.update({'a0': a})
                self.paraset.update({'b0': b})
                self.paraset.update({'c0': c})
                self.paraset.update({'d0': d})

        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
                if self.ntype != 'LIF':
                    if self.ntype=='Pretrain':
                        a, b, c, d = self.generate_parasets(hidden_sizes[layer])
                        self.paraset.update({'a' + str(layer): a})
                        self.paraset.update({'b' + str(layer): b})
                        self.paraset.update({'c' + str(layer): c})
                        self.paraset.update({'d' + str(layer): d})

                    else:
                        print(self.ntype)
                        a, b, c, d = self.generate_parasets(hidden_sizes[layer])
                        self.paraset.update({'a' + str(layer): a})
                        self.paraset.update({'b' + str(layer): b})
                        self.paraset.update({'c' + str(layer): c})
                        self.paraset.update({'d' + str(layer): d})

        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)
        if self.ntype != 'LIF':
            if self.ntype=='Pretrain':
                a, b, c, d = self.generate_parasets(out_pop_dim)
                self.paraset.update({'a' + str(self.hidden_num): a})
                self.paraset.update({'b' + str(self.hidden_num): b})
                self.paraset.update({'c' + str(self.hidden_num): c})
                self.paraset.update({'d' + str(self.hidden_num): d})
            else:
                a, b, c, d = self.generate_parasets(out_pop_dim)
                self.paraset.update({'a' + str(self.hidden_num): a})
                self.paraset.update({'b' + str(self.hidden_num): b})
                self.paraset.update({'c' + str(self.hidden_num): c})
                self.paraset.update({'d' + str(self.hidden_num): d})


    def generate_parasets(self,output_size):
        if self.ntype!='Pretrain':
            paraset = make_paraset(self.ntype)
            lens = len(paraset)
            org = lens * torch.rand((output_size))
            org = org.floor()
            org = org.clamp(0, lens - 1)
            num = []
            for i in range(lens):
                num.append(0)

            for i in range(output_size):
                num[int(org[i].data)] += 1

            a = None
            b = None
            c = None
            d = None

            for i in range(lens):
                if a is None:
                    if num[i] != 0:
                        a = paraset[i][0] * torch.ones((1, num[i]))
                        b = paraset[i][1] * torch.ones((1, num[i]))
                        c = paraset[i][2] * torch.ones((1, num[i]))
                        d = paraset[i][3] * torch.ones((1, num[i]))
                else:
                    if num[i] != 0:
                        a = torch.cat((a, paraset[i][0] * torch.ones((1, num[i]))), 1)
                        b = torch.cat((b, paraset[i][1] * torch.ones((1, num[i]))), 1)
                        c = torch.cat((c, paraset[i][2] * torch.ones((1, num[i]))), 1)
                        d = torch.cat((d, paraset[i][3] * torch.ones((1, num[i]))), 1)
            a = a.to(self.device)
            b = b.to(self.device)
            c = c.to(self.device)
            d = d.to(self.device)

            return a,b,c,d
        elif self.ntype=='Pretrain':

            a, b, c, d = [0.02, 0.2, 0, 0.08]  # 固定初始化
            #import numpy as np
            #a,b,c,d = np.random.uniform(0,1,size=4).tolist()  # 0-1 均匀随机初始化
            #a,b,c,d = np.random.uniform(-0.5,0.5,size=4).tolist()  # -0.5 - 0.5 均匀随机初始化

            a = nn.Parameter(a * torch.ones((1,output_size)))
            b = nn.Parameter(b * torch.ones((1,output_size)))
            c = nn.Parameter(c * torch.ones((1,output_size)))
            d = nn.Parameter(d * torch.ones((1,output_size)))
            distrue = True
            a.requires_grad = distrue
            b.requires_grad = distrue
            c.requires_grad = distrue
            d.requires_grad = distrue

            return a,b,c,d

    def neuron_model_LIF(self, syn_func, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def neuron_model_MDN(self, syn_func, pre_layer_output, current, volt, spike,u, layer):
        volt = volt*(1-spike) + spike*self.paraset['c'+str(layer)]  #
        #  membrane recovery variable
        u = u + spike*self.paraset['d'+str(layer)]
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)  #

        volt_delta = volt*volt - volt - u +current
        u_delta = self.paraset['a'+str(layer)]*(self.paraset['b'+str(layer)]*volt - u)

        volt = volt + volt_delta
        u = u + u_delta

        spike = self.pseudo_spike(volt)
        return current, volt, spike,u



    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        if self.ntype=='LIF':
            # Define LIF Neuron states: Current, Voltage, and Spike
            hidden_states = []
            for layer in range(self.hidden_num):
                hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                      for _ in range(3)])
            out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                              for _ in range(3)]
            out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
            # Start Spike Timestep Iteration
            for step in range(self.spike_ts):
                in_pop_spike_t = in_pop_spikes[:, :, step]
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model_LIF(
                    self.hidden_layers[0], in_pop_spike_t,
                    hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
                )
                if self.hidden_num > 1:
                    for layer in range(1, self.hidden_num):
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model_LIF(
                            self.hidden_layers[layer], hidden_states[layer-1][2],
                            hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                        )
                out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model_LIF(
                    self.out_pop_layer, hidden_states[-1][2],
                    out_pop_states[0], out_pop_states[1], out_pop_states[2]
                )
                out_pop_act += out_pop_states[2]
            out_pop_act = out_pop_act / self.spike_ts
        else:
            # Define 二阶动力学 Neuron states: Current, Voltage, Spike and U
            hidden_states = []
            for layer in range(self.hidden_num):
                hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                      for _ in range(3)] + [
                                         0.08 * torch.ones(batch_size, self.hidden_sizes[layer], device=self.device)])

            out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                              for _ in range(3)] + [
                                 0.08 * torch.ones(batch_size, self.out_pop_dim, device=self.device)]

            out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)

            # Start Spike Timestep Iteration
            for step in range(self.spike_ts):
                in_pop_spike_t = in_pop_spikes[:, :, step]
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][
                    3] = self.neuron_model_MDN(
                    self.hidden_layers[0], in_pop_spike_t,
                    hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][3], 0
                )
                if self.hidden_num > 1:
                    for layer in range(1, self.hidden_num):
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][
                            2], hidden_states[layer][3] = self.neuron_model_MDN(
                            self.hidden_layers[layer], hidden_states[layer - 1][2],
                            hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2],
                            hidden_states[layer][3], layer
                        )
                out_pop_states[0], out_pop_states[1], out_pop_states[2], out_pop_states[3] = self.neuron_model_MDN(
                    self.out_pop_layer, hidden_states[-1][2],
                    out_pop_states[0], out_pop_states[1], out_pop_states[2], out_pop_states[3], self.hidden_num
                )
                out_pop_act += out_pop_states[2]

            out_pop_act = out_pop_act / self.spike_ts

        return out_pop_act

class SpikeActor(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder """
    def __init__(self, state_dim, action_dim,max_action, hidden_sizes=[256, 256],encoder_pop_dim=10, decoder_pop_dim=10,
                 mean_range=(-3,3), std=math.sqrt(0.15), spike_ts=10, device=torch.device('cpu'),encoder='pop', to_spike='none',ntype='LIF',actorLR=1e-5):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        """
        super().__init__()
        self.max_action = max_action
        print("Spike Actor!")

        if encoder=='pop':
            if to_spike == 'poisson':
                self.encoder = PopSpikeEncoderPoissonSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std,
                                                           device)
            elif to_spike == 'regular':
                self.encoder = PopSpikeEncoderRegularSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std,
                                                           device)
            elif to_spike == 'uniform':
                self.encoder = PopSpikeEncoderUniformSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std,
                                                           device)
            elif to_spike == 'none':
                self.encoder = PopSpikeEncoderNoneSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std, device)
        elif encoder=='spine':
            if to_spike == 'poisson':
                self.encoder = SpineSpikeEncoderPoissonSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std,
                                                           device)
            elif to_spike == 'regular':
                self.encoder = SpineSpikeEncoderRegularSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std,
                                                           device)
            elif to_spike == 'uniform':
                self.encoder = SpineSpikeEncoderUniformSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std,
                                                           device)
            elif to_spike == 'none':
                self.encoder = SpineSpikeEncoderNoneSpike(state_dim, encoder_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(state_dim*encoder_pop_dim, action_dim*decoder_pop_dim, hidden_sizes, spike_ts, device,ntype)
        self.decoder = PopSpikeDecoder(action_dim, decoder_pop_dim)


    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        batch_size = obs.shape[0]
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)

        action = self.max_action * self.decoder(out_pop_activity)

        return action
