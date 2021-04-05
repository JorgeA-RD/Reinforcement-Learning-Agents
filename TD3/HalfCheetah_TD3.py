import gym
import pybullet_envs
from TD3_Agent import TD3_Agent

#Environment Parameters
env = gym.make('HalfCheetahBulletEnv-v0')
num_states = 26
num_actions= 6
max_steps = 999
#reward_threshold=3000
action_low = [-1,-1,-1,-1,-1,-1]
action_high = [1,1,1,1,1,1]


#Agent Hyperparameters
critic_hidden_units = [400,300]
actor_hidden_units = [400,300]

gamma = 0.99
eta_critic = 1e-3
eta_actor = 1e-3

gauss_reg = 0.2
clip_reg = 0.5
delayed_train = 2

train_after = 10000
train_every = 1000
gradient_steps = 1000

batch_size = 100
cap_buffer = 1000000
episodes = 2000

soft_tau = 0.005
gauss_scale = 0.1

Agent = TD3_Agent(num_states, actor_hidden_units, critic_hidden_units, num_actions, 
                 action_low, action_high, gamma, eta_critic, eta_actor, soft_tau, 
                 batch_size, cap_buffer, gauss_reg, clip_reg, delayed_train, 
                 train_after, train_every, gradient_steps)
#####################################################################
#####################################################################

Agent.init_weights()
t = 0

for episod in range(episodes):
    
    observation = env.reset()
    k=0
    T_Rew=0
    
    while True:
    
        action =Agent.action_selection(observation,gauss_scale)
        next_observation, reward, done,_ = env.step(action)
        Agent.fill_replay_buff(observation, action, reward, next_observation, done, k, max_steps)
        Agent.train(t)
        observation = next_observation
        k+=1
        t+=1
        T_Rew += reward
        
        if len(Agent.rep_buffer['s']) < cap_buffer and len(Agent.rep_buffer['s']) % 10000 == 0:
            print('Replay buffer ',len(Agent.rep_buffer['s']),' of ', cap_buffer)
            
        if done:
            Agent.fill_history(T_Rew)
            break;
    
    if t >= train_after:        
        print('Ep ',episod, 'TR', Agent.history['Tot_Rew'][-1], 'AR ',Agent.history['Avg_Rew'][-1])

env.close()

Agent.actor_train.save_weights('./Saved_Weights/halfcheetah_td3_bn')
Agent.plot_results()
Agent.save_history("halfcheetah_td3_64_hist.data")
