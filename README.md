# Spine-SNN Actor network based on TD3

PyTorch implementation of TD3 with spine-SNN actor network. The code based on the paper "Spine encoding improved spiking neural network for efficient reinforcement learning".

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.2](https://github.com/pytorch/pytorch) and Python 3.7. 

We use the code TD3 of Fujimoto2018 [paper](https://arxiv.org/abs/1802.09477).

### Usage
Experiments on single environments can be run by calling:
```
python main.py --env=HalfCheetah-v2 --network=EpspAdeepC
```

Hyper-parameters can be modified with different arguments to main.py. The configuration about the experiments is outline in the paper.

### Results
Experiment in the paper such as the influence of spine encoder dim and the comparison between different actor network could reproduce with this code.
