import gym
from DQN_Agent import DDQN_Agent

#Environment Parameters
env = gym.make('CartPole-v1')
num_states = 4
num_actions= 1
action_space_dim = [2]
max_steps = 499

#Agent Hyperparameters
hidden_units = [128,128]

gamma = 0.99
eta = 1e-4
delta_huber = 5
epsilon_init = 0.6
epsilon_fin = 0.01
soft_tau = 0.001

cap_buffer = 5e5
batch_size = 64
episodes = 52000


Agent = DDQN_Agent(num_states, hidden_units, num_actions, action_space_dim, 
                 batch_size,gamma,cap_buffer,eta,delta_huber)


#####################################################################
#####################################################################

#c_ext = 0

for episod in range(episodes):
    
    observation = env.reset()
    done = False
    k=0
    T_Rew=0
    
    epsilon = epsilon_init + ((epsilon_fin-epsilon_init)/ episodes)*episod
    
    while not done:
        
        if len(Agent.rep_buffer['s']) == cap_buffer and episod % 15 == 0:
            env.render()
        
        action = Agent.action_selection(observation,epsilon)
        next_observation, reward, done,_ = env.step(action[0])
        Agent.fill_replay_buff(observation,action,reward,next_observation,done,k,max_steps)
        observation = next_observation
        
        loss = Agent.train()
        
        k+=1
        #c_ext += 1
        T_Rew += reward
        
        if len(Agent.rep_buffer['s']) == cap_buffer:
            Agent.copy_weights(soft_tau)
        #if len(Agent.rep_buffer['s']) == cap_buffer and c_ext % 80 == 0:
            #Agent.copy_weights_normal()
            #c_ext = 0
        
        if len(Agent.rep_buffer['s']) == cap_buffer and done:
            print('Episode: ', episod, ' Tot Rew: ', T_Rew, ' Loss: ', loss)
            
        if len(Agent.rep_buffer['s']) < cap_buffer and len(Agent.rep_buffer['s']) % 10000 == 0:
            print('Replay buffer ',len(Agent.rep_buffer['s']),' of ', cap_buffer)
            
env.close()