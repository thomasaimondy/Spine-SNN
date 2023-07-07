
import math

import torch
import torch.nn as nn
from utils import make_paraset
import copy
"""
Parameters for SNN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
Epsp_PSEUDO_GRAD_WINDOW = 0.5

#群编码有参数 所以需要写伪BP
class PseudoEncoderEpspRegular(torch.autograd.Function):
    """ Pseudo-gradient function for Epsp - Regular Epsp for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PseudoEncoderEpspPoisson(torch.autograd.Function):
    """ Pseudo-gradient function for Epsp - Poisson Epsp for encoder """
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PseudoEncoderEpspUniform(torch.autograd.Function):
    """ Pseudo-gradient function for Epsp - Poisson Epsp for encoder """
    @staticmethod
    def forward(ctx, input):  #input (batch_size,obs_dim*spine_dim)
        return (input > torch.rand(size=input.shape,device=input.device)).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class SpineEpspEncoderNoneEpsp(nn.Module):
    """ Learnable spine Coding Epsp Encoder with Regular Epsp Trains """
    def __init__(self, obs_dim, spine_dim, Epsp_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param spine_dim: spine dimension
        :param Epsp_ts: Epsp timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.spine_dim = spine_dim
        self.encoder_neuron_num = obs_dim * spine_dim
        self.Epsp_ts = Epsp_ts
        self.device = device
        self.pseudo_Epsp = PseudoEncoderEpspRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, spine_dim)
        if spine_dim>1:
            delta_mean = (mean_range[1] - mean_range[0]) / (spine_dim - 1)
            for num in range(spine_dim):
                tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        elif spine_dim==1:
            tmp_mean[0, :, 0] = 0
        tmp_std = torch.zeros(1, obs_dim, spine_dim) + std
        self.mean_encoder = nn.Parameter(tmp_mean)
        self.std_encoder = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: spine_Epsps
        """
        obs = obs.view(-1, self.obs_dim, 1)  # (batch_size,obs_dim,1)
        # Receptive Field of spine encoder has Sigmoid Shape
        spine_act=1/(1+torch.exp(-(obs-self.mean_encoder)/self.std_encoder)).view(-1, self.encoder_neuron_num)
        spine_Epsps = torch.zeros(batch_size, self.encoder_neuron_num, self.Epsp_ts,
                                 device=self.device)  # (batch_size,obs_dim*spine_dim,Epsp_ts)
        # Generate Poisson Epsp Trains
        for step in range(self.Epsp_ts):
            # spine_Epsps[:, :, step] = self.pseudo_Epsp(spine_act)
            spine_Epsps[:, :, step] = spine_act
        return spine_Epsps


class SpineEpspDecoder(nn.Module):
    """ Spine Coding Epsp Decoder """
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

class EpspDecoder(nn.Module):
    """ Population Coding Epsp Decoder """
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

class PseudoEpspRect(torch.autograd.Function):
    """ Pseudo-gradient function for Epsp - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        Epsp_pseudo_grad = (abs(input - NEURON_VTH) < Epsp_PSEUDO_GRAD_WINDOW)
        return grad_input * Epsp_pseudo_grad.float()


class EpspMLP(nn.Module):
    """ Epsp MLP with Input and Output neurons """
    def __init__(self, in_dim, out_dim, hidden_sizes, Epsp_ts, device, ntype):
        """
        :param in_dim: input population dimension
        :param out_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param Epsp_ts: Epsp timesteps
        :param device: device
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.Epsp_ts = Epsp_ts
        self.device = device
        self.ntype = ntype

        self.pseudo_Epsp = PseudoEpspRect.apply
        print(ntype)

        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_dim, hidden_sizes[0])])
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

        self.out_layer = nn.Linear(hidden_sizes[-1], out_dim)
        if self.ntype != 'LIF':
            if self.ntype=='Pretrain':
                a, b, c, d = self.generate_parasets(out_dim)
                self.paraset.update({'a' + str(self.hidden_num): a})
                self.paraset.update({'b' + str(self.hidden_num): b})
                self.paraset.update({'c' + str(self.hidden_num): c})
                self.paraset.update({'d' + str(self.hidden_num): d})
            else:
                a, b, c, d = self.generate_parasets(out_dim)
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

    def neuron_model_LIF(self, syn_func, pre_layer_output, current, volt, Epsp):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param Epsp: Epsp of last step
        :return: current, volt, Epsp
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - Epsp) + current
        Epsp = self.pseudo_Epsp(volt)
        return current, volt, Epsp

    def neuron_model_MDN(self, syn_func, pre_layer_output, current, volt, Epsp,u, layer):
        volt = volt*(1-Epsp) + Epsp*self.paraset['c'+str(layer)]  #
        #  membrane recovery variable
        u = u + Epsp*self.paraset['d'+str(layer)]
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)  #

        volt_delta = volt*volt - volt - u +current
        u_delta = self.paraset['a'+str(layer)]*(self.paraset['b'+str(layer)]*volt - u)

        volt = volt + volt_delta
        u = u + u_delta

        Epsp = self.pseudo_Epsp(volt)
        return current, volt, Epsp,u



    def forward(self, in_Epsps, batch_size):
        """
        :param in_Epsps: input population Epsps
        :param batch_size: batch size
        :return: out_act
        """
        if self.ntype=='LIF':
            # Define LIF Neuron states: Current, Voltage, and Epsp
            hidden_states = []
            for layer in range(self.hidden_num):
                hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                      for _ in range(3)])
            out_states = [torch.zeros(batch_size, self.out_dim, device=self.device)
                          for _ in range(3)]
            out_act = torch.zeros(batch_size, self.out_dim, device=self.device)
            # Start Epsp Timestep Iteration
            for step in range(self.Epsp_ts):
                in_Epsp_t = in_Epsps[:, :, step]
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model_LIF(
                    self.hidden_layers[0], in_Epsp_t,
                    hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
                )
                if self.hidden_num > 1:
                    for layer in range(1, self.hidden_num):
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model_LIF(
                            self.hidden_layers[layer], hidden_states[layer-1][2],
                            hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                        )
                out_states[0], out_states[1], out_states[2] = self.neuron_model_LIF(
                    self.out_layer, hidden_states[-1][2],
                    out_states[0], out_states[1], out_states[2]
                )
                out_act += out_states[2]
            out_act = out_act / self.Epsp_ts
        else:
            # Define 二阶动力学 Neuron states: Current, Voltage, Epsp and U
            hidden_states = []
            for layer in range(self.hidden_num):
                hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                      for _ in range(3)] + [
                                         0.08 * torch.ones(batch_size, self.hidden_sizes[layer], device=self.device)])

            out_states = [torch.zeros(batch_size, self.out_dim, device=self.device)
                          for _ in range(3)] + [
                                 0.08 * torch.ones(batch_size, self.out_dim, device=self.device)]

            out_act = torch.zeros(batch_size, self.out_dim, device=self.device)

            # Start Epsp Timestep Iteration
            for step in range(self.Epsp_ts):
                in_Epsp_t = in_Epsps[:, :, step]
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][
                    3] = self.neuron_model_MDN(
                    self.hidden_layers[0], in_Epsp_t,
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
                out_states[0], out_states[1], out_states[2], out_states[3] = self.neuron_model_MDN(
                    self.out_layer, hidden_states[-1][2],
                    out_states[0], out_states[1], out_states[2], out_states[3], self.hidden_num
                )
                out_act += out_states[2]

            out_act = out_act / self.Epsp_ts

        return out_act

class EpspActor(nn.Module):
    """ Population Coding Epsp Actor with Fix Encoder """
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256], encoder_spine_dim=10, decoder_dim=10,
                 mean_range=(-3,3), std=math.sqrt(0.15), Epsp_ts=10, device=torch.device('cpu'), encoder='pop', to_Epsp='none', ntype='LIF', actorLR1=1e-5, actorLR2=1e-5, rate=10):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_spine_dim: spine encoding dimension
        :param dep_dim: decoder dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param Epsp_ts: Epsp timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson Epsps for encoder
        """
        super().__init__()
        self.max_action = max_action
        print("Epsp Actor!")
        if encoder=='spine':
            self.encoder = SpineEpspEncoderNoneEpsp(state_dim, encoder_spine_dim, Epsp_ts, mean_range, std, device)
        self.snn = EpspMLP(state_dim * encoder_spine_dim, action_dim * decoder_dim, hidden_sizes, Epsp_ts, device, ntype)
        self.decoder = EpspDecoder(action_dim, decoder_dim)


    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        batch_size = obs.shape[0]
        in_Epsps = self.encoder(obs, batch_size)
        out_activity = self.snn(in_Epsps, batch_size)

        action = self.max_action * self.decoder(out_activity)

        return action

    def mini(self):
        self.encoder.mini()