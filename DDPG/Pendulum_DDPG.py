import gym
from DDPG_Agent import DDPG_Agent

#Environment Parameters
env = gym.make('Pendulum-v0')
num_states = 3
num_actions= 1
max_steps = 199
action_low = [-2]
action_high = [2]


#Agent Hyperparameters
critic_hidden_units = [128,128]
actor_hidden_units = [128,128]

gamma = 0.99
eta_critic = 3e-3
eta_actor = 3e-4
delta_huber = 1

batch_size = 64
cap_buffer = 1e6
episodes = 20000

soft_tau = 0.001
gauss_scale = 1.5
gauss_decay = 0.9997 

Agent = DDPG_Agent(num_states,actor_hidden_units,critic_hidden_units,num_actions,
                 action_low,action_high,gamma,eta_critic,eta_actor,delta_huber,batch_size,cap_buffer)                                            

#####################################################################
#####################################################################

for episod in range(episodes):
    
    observation = env.reset()
    k=0
    T_Rew=0
    
    #gauss_scale = gauss_init+((gauss_fin-gauss_init)/(episodes))*episod
    gauss_scale = gauss_scale*gauss_decay
    
    while True:
    
        action =Agent.action_selection(observation,gauss_scale)
        if episod % 10 == 0 and len(Agent.rep_buffer['s']) == cap_buffer:    
            env.render()
        
        next_observation, reward, done,_ = env.step(action)
        Agent.fill_replay_buff(observation, action, reward, next_observation, done, k, max_steps)
        critic_loss, actor_loss = Agent.train()
        observation = next_observation
        k+=1
        T_Rew += reward
        
        if len(Agent.rep_buffer['s']) == cap_buffer:
            Agent.copy_weights(soft_tau)
            
            if done:
                print('Ep ',episod, 'Critic Loss ',critic_loss.numpy(),'Actor Loss ',actor_loss.numpy(), 
                      f'Tot_Rew {T_Rew:.2f}')
        
        if len(Agent.rep_buffer['s']) < cap_buffer and k % 199 == 0:
            print('Replay buffer ',len(Agent.rep_buffer['s']),' of ', cap_buffer)
        
        if done:
            break;

env.close()