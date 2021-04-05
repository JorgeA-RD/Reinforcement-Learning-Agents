import gym
from TD3_Agent import TD3_Agent

#Environment Parameters
env = gym.make('Pendulum-v0')
num_states = 3
num_actions= 1
max_steps = 199
action_low = [-2]
action_high = [2]


#Agent Hyperparameters
critic_hidden_units = [64,64]
actor_hidden_units = [64,64]

gamma = 0.99
eta_critic = 3e-3
eta_actor = 3e-3

gauss_reg = 0.2
clip_reg = 0.5
delayed_train = 2

train_after = 1000
train_every = 100
gradient_steps = 100

batch_size = 128
cap_buffer = 50000
episodes = 500

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
        if episod % 20 == 0 and t >= train_after:    
            env.render()
        
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

#Agent.actor_train.save_weights('./Saved_Weights/pendulum_td3_2')
Agent.plot_results()
#Agent.save_history("pendulum_hist_2.data")
