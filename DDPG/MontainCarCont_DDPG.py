import gym
from DDPG_Agent import DDPG_Agent

#Environment Parameters
env = gym.make('MountainCarContinuous-v0')
num_states = 2
num_actions = 1
max_steps = 999                 
action_low = [-1]
action_high = [1]


#Agent Hyperparameters
critic_hidden_units = [128,128]
actor_hidden_units = [128,128]

gamma = 0.99
eta_critic = 0.001
eta_actor = 0.0001
delta_huber = 1

batch_size = 64
cap_buffer = 100000
episodes = 700

soft_tau = 0.001
gauss_init = 1.5
gauss_fin = 0.1

Agent=DDPG_Agent(num_states,actor_hidden_units,critic_hidden_units,num_actions,
                 action_low,action_high,gamma,eta_critic,eta_actor,delta_huber,batch_size,cap_buffer)


#####################################################################
#####################################################################

for episod in range(episodes):
  
    
    
    observation = env.reset()
    k=0
    T_Rew=0
    
    gauss_scale = gauss_init+((gauss_fin-gauss_init)/(episodes))*episod
    
    while True:
    
        action =Agent.action_selection(observation,gauss_scale)
        if episod % 10 == 0 and len(Agent.rep_buffer['s']) == cap_buffer:    
            env.render()
        
        next_observation, reward, done,_ = env.step(action)
        Agent.fill_replay_buff(observation, action, reward, next_observation, done, k, max_steps)
        critic_loss, actor_loss=Agent.train()
        observation = next_observation
        k+=1
        T_Rew += reward
        
        if len(Agent.rep_buffer['s']) == cap_buffer:
            Agent.copy_weights(soft_tau)
            
            succ = bool(observation[0] >= 0.45 and observation[1] >= 0)
            if done:
                print('Critic Loss ',critic_loss.numpy(),'Actor Loss ',actor_loss.numpy(),'Ep ',episod,
                      'Step',k,'Sucsesfull ',succ, f'Tot_Rew {T_Rew:.2f}')
        
        if len(Agent.rep_buffer['s']) < cap_buffer and k % 999 == 0:
            print('Replay buffer ',len(Agent.rep_buffer['s']),' of ', cap_buffer)
        
        if done:
            break;
    
env.close()