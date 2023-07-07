
import gym
from utils import *
from TD3 import TD3
import math
import time
import random


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10): # 默认测试10个轨迹
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100) # seed + 100 与训练环境区分

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state)) # 测试是只需要 actor 即policy
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward # 所有轨迹的总reward

	avg_reward /= eval_episodes # 平均每个轨迹的reward和

	return avg_reward

def Epsp_td3(env_name,memo, ac_kwargs=dict(),seed=0, tb_comment='', model_idx=0,network='deepAC',start_timesteps=25e3,
				  eval_freq=1e4,max_timesteps=2e6,expl_noise=0.1,
				  batch_size=256,discount=0.99,tau=0.005,policy_noise=0.2,
				  noise_clip=0.5,policy_freq=2):


	env = gym.make(env_name)

	# Set seeds  固定gym torch numpy的随机种子
	env.seed(seed)
	env.action_space.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0]) # act_limit


	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": discount, # gamma 0.99
		"tau": tau,  #0.005     1-polyak
	}

	# Initialize policy

	# Target policy smoothing is scaled wrt the action scale
	kwargs["policy_noise"] = policy_noise * max_action
	kwargs["noise_clip"] = noise_clip * max_action
	kwargs["policy_freq"] = policy_freq


	policy = TD3(network,**kwargs,ac_kwargs=ac_kwargs)



	replay_buffer = ReplayBuffer(state_dim, action_dim)
	save_test_reward = [eval_policy(policy, env_name, seed)] # 训练前policy的测试结果
	save_test_reward_steps = [0]
	print("Model: ", model_idx, " Steps: ", save_test_reward_steps[0], " Mean Reward: ", save_test_reward[0])


	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	best_test_mean_reward = -5000
	time1 = time.time()
	for t in range(int(max_timesteps)): # 200w

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < start_timesteps: # 25000
			action = env.action_space.sample()
		else:
			action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * expl_noise, size=action_dim) #训练中会对action添加噪声，测试时不会，直接用actor网络的输出
			).clip(-max_action, max_action) # 缩放到动作的范围

		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= start_timesteps:
			policy.train(replay_buffer, batch_size)

		if done: #一条轨迹终止 重置环境 进入下一条轨迹 总共交互200w步
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			#print(
				#f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % eval_freq == 0:  # 每1w步 评估一次当前的policy
			test_mean_reward = eval_policy(policy, env_name, seed)
			save_test_reward.append(test_mean_reward) # 100w/1w +1 = 201 个值  包含第0步测试的值
			save_test_reward_steps.append(t+1)
			print("Model: ", model_idx, " Steps: ", t + 1, " Mean Reward: ", test_mean_reward)
			memo.write("Model: "+ str(model_idx)+ " Steps: "+str( t + 1)+ " Mean Reward: "+str( test_mean_reward)+'\n')


			if args.ntype == 'Pretrain' and network == 'EpspAdeepC':

				if test_mean_reward > best_test_mean_reward:
					best_test_mean_reward = test_mean_reward
					'''
					torch.save(policy.actor.snn.paraset, model_dir_pretrain + '/model' + str(model_idx) + '_paraset.pth')
                    '''
			time2 = time.time()
			print("spend time of 5k steps:",time2-time1)
			memo.write("spend time of 5k steps:"+str(time2-time1)+'\n')
			memo.flush()
			time1 = time2

	# Save Test Reward List
	'''
	pickle.dump([save_test_reward, save_test_reward_steps],
				open(model_dir + '/' + "model" + str(model_idx) + "_test_rewards.p", "wb+"))
    '''

if __name__ == "__main__":
	'''
	 env                         obs_dim             act_dim    
	 HalfCheetah-v3              17,                 6,
	 Walker2d-v3                 17,                 6,
	 Hopper-v3                   11,                 3,
	 Swimmer-v3                  8,                  2,
	 '''
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Swimmer-v3")          # OpenAI gym environment name
	parser.add_argument('--network', type=str, default='deepAC_spine', help=['deepAC', 'deepAC_spine','EpspAdeepC'])
	parser.add_argument('--start_model_idx', type=int, default=0)
	parser.add_argument('--num_model', type=int, default=20)  # 多个模型/多个随机种子 下运行  结果取平均           预训练 在一个环境上、一个seed下、聚类一个中心得到 四个动力学参数

	parser.add_argument('--encoder_spine_dim', type=int, default=20)
	parser.add_argument('--encoder', type=str, default='spine')  #spine or none
	parser.add_argument('--Epsp_ts', type=int, default=5)    # 5->10 能够更好的体现二阶神经元的动力学优势
	parser.add_argument('--ntype', type=str, default='A1')  # LIF,A1
	parser.add_argument('--actorLR1', type=float, default=1e-5)
	parser.add_argument('--actorLR2', type=float, default=1e-5)

	parser.add_argument("--start_timesteps", default=2e4, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1000000, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates

	args = parser.parse_args()

	print(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("device:", device)

	if args.network == 'EpspAdeepC':
		AC_KWARGS = dict(hidden_sizes=[256, 256],
						 encoder_spine_dim=args.encoder_spine_dim,
						 decoder_dim=10,
						 mean_range=(-3, 3),
						 std=math.sqrt(0.15),
						 Epsp_ts=args.Epsp_ts,
						 device=device,
						 encoder=args.encoder,
						 ntype=args.ntype,
						 actorLR1=args.actorLR1,
						 actorLR2=args.actorLR2
						 )
		COMMENT = "td3-EpspAdeepC-" + args.env_name + '-' + '-Epsp_ts-' + str(args.Epsp_ts)+\
				  '-encoder-' + str(args.encoder) + "-encoder-dim-" + str(AC_KWARGS['encoder_spine_dim'])+'-' + args.ntype

	elif args.network == 'deepAC':
		AC_KWARGS = dict(hidden_sizes=[256, 256],
						 actorLR1=args.actorLR1,
						 actorLR2=args.actorLR2
						 )
		COMMENT = "td3-deepAC-" + args.env_name
	elif args.network == 'deepAC_spine':
		AC_KWARGS = dict(hidden_sizes=[256, 256],
						 encoder_spine_dim=10,
						 decoder_dim=10,
						 mean_range=(-3, 3),
						 std=math.sqrt(0.15),
						 device=device,
						 actorLR1=args.actorLR1,
						 actorLR2=args.actorLR2
						 )
		COMMENT = "td3-deepAC_spine-" + args.env_name

	print(AC_KWARGS)
	f=open(file=args.env_name+'_'+args.encoder+'_'+args.ntype+'_'+str(args.actorLR1)+'_'+str(args.actorLR2)+'_'+str(args.encoder_spine_dim)+'_'+str(args.start_model_idx)+'_'+str(args.Epsp_ts)+'.txt',mode='w')
	for num in range(args.start_model_idx, args.start_model_idx + args.num_model): # 把多个模型的循环放在py脚本内
		print("model: ", num)
		#seed = num * 10
		random_seed = num

		Epsp_td3(args.env_name,f, ac_kwargs=AC_KWARGS,
				 seed=random_seed, tb_comment=COMMENT, model_idx=num,network=args.network,start_timesteps=args.start_timesteps,
				  eval_freq=args.eval_freq,max_timesteps=args.max_timesteps,expl_noise=args.expl_noise,
				  batch_size=args.batch_size,discount=args.discount,tau=args.tau,policy_noise=args.policy_noise,
				  noise_clip=args.noise_clip,policy_freq=args.policy_freq
				  )



