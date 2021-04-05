import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pickle

#policy network
class actor_netw(tf.keras.Model):
    
    def __init__(self,num_states, actor_hidden_units, num_actions, action_high):
        super(actor_netw,self).__init__()
        
        self.action_high = action_high
        
        self.actor_input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        
        self.actor_hidden_layers = []
        
        for i in actor_hidden_units:
            self.actor_hidden_layers.append(
                    tf.keras.layers.Dense(i, activation= tf.nn.leaky_relu, kernel_initializer= 'RandomNormal'))
                    
        self.actor_output_layer = tf.keras.layers.Dense(num_actions, activation= 'tanh', kernel_initializer= 'RandomNormal')

    @tf.function
    def call(self,inputs):
        
        z = self.actor_input_layer(inputs)
        
        for layer in self.actor_hidden_layers:
            z = layer(z)
        output = self.actor_output_layer(z)*self.action_high
        
        return output
    
#q network
class critic_netw(tf.keras.Model):
    
    def __init__(self,num_states, critic_hidden_units, num_actions):
        super(critic_netw,self).__init__()
        
        
        #Q1
        self.critic1_input_layer = tf.keras.layers.InputLayer(input_shape=(num_states + num_actions,))
        
        self.critic1_hidden_layers = []
        
        for j in critic_hidden_units:
            self.critic1_hidden_layers.append(
                    tf.keras.layers.Dense(j, activation= tf.nn.leaky_relu, kernel_initializer= 'RandomNormal'))
                    
        self.critic1_output_layer = tf.keras.layers.Dense(1, activation= 'linear', kernel_initializer= 'RandomNormal')
        
        
        #Q2
        self.critic2_input_layer = tf.keras.layers.InputLayer(input_shape=(num_states + num_actions,))
        
        self.critic2_hidden_layers = []
        
        for j in critic_hidden_units:
            self.critic2_hidden_layers.append(
                    tf.keras.layers.Dense(j, activation= tf.nn.leaky_relu, kernel_initializer= 'RandomNormal'))
                    
        self.critic2_output_layer = tf.keras.layers.Dense(1, activation= 'linear', kernel_initializer= 'RandomNormal')

    @tf.function
    def call(self,input_state,input_action):
        
        inputs=tf.concat((input_state,input_action),1)
        
        #Q1
        z1 = self.critic1_input_layer(inputs)
        
        for layer in self.critic1_hidden_layers:
            z1 = layer(z1)
        
        output_c1 = self.critic1_output_layer(z1)
        
        #Q2
        z2 = self.critic2_input_layer(inputs)
        
        for layer in self.critic2_hidden_layers:
            z2 = layer(z2)
        
        output_c2 = self.critic2_output_layer(z2)
        
        return output_c1, output_c2
    
class TD3_Agent:
    
    def __init__(self,num_states, actor_hidden_units, critic_hidden_units, num_actions, 
                 action_low, action_high, gamma, eta_critic, eta_actor, soft_tau, 
                 batch_size, cap_buffer, gauss_reg, clip_reg, delayed_train, 
                 train_after, train_every, gradient_steps):
        
        self.num_states=num_states
        self.num_actions=num_actions
        self.action_low=action_low
        self.action_high=action_high
        
        self.gamma=gamma
        self.critic_optimizer = tf.optimizers.Adam(eta_critic)
        self.actor_optimizer = tf.optimizers.Adam(eta_actor)
        self.batch_size=batch_size
        self.soft_tau = soft_tau
        
        self.gauss_reg = gauss_reg
        self.clip_reg = clip_reg
        self.delayed_train = delayed_train
        
        self.train_after = train_after 
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        
        
        self.history = {'Tot_Rew':[], 'Avg_Rew':[]}
        self.rep_buffer = {'s':[],'a':[],'r':[],'sn':[],'done':[]}
        self.cap_buffer = cap_buffer
        
        self.actor_train=actor_netw(num_states, actor_hidden_units, num_actions, action_high)
        self.actor_target=actor_netw(num_states, actor_hidden_units, num_actions, action_high)
        self.actor_target.trainable=False
        
        self.critic_train=critic_netw(num_states, critic_hidden_units, num_actions)  
        self.critic_target=critic_netw(num_states, critic_hidden_units, num_actions)
        self.critic_target.trainable=False
    
    
    
    def action_selection(self,state,gauss_scale):
 
        noise=np.float32(np.random.normal(0,gauss_scale,self.num_actions))
        action=self.actor_train(np.asarray([state],dtype=np.float32))
        
        action=np.reshape(action,self.num_actions) + noise
        action=np.clip(action,np.float32(self.action_low),np.float32(self.action_high))
        
        return action.tolist()
    
        
    def fill_replay_buff(self, state, action, reward, next_state, done, step, max_steps):
        #Avoid confuising a final state in an episode trajectory with the terminal state
        if done and step < max_steps:
            done_dummy =  True 
        else: 
            done_dummy = False 
        
        experience = {'s':state,'a':action,'r':reward,'sn':next_state,'done':done_dummy}
        
        if len(self.rep_buffer['s']) == self.cap_buffer:
            for key in self.rep_buffer.keys():
                self.rep_buffer[key].pop(0)
        for key, value in experience.items():
            self.rep_buffer[key].append(value)
            
            
    def fill_history(self,T_Rew):
        
        T_Rew = np.float32(T_Rew)
        if len(self.history['Tot_Rew']) >= 50:
            avg_rew = sum(self.history['Tot_Rew'][-50:])/50
            avg_rew = np.float32(avg_rew)
        else:
            avg_rew = None
        
        self.history['Tot_Rew'].append(T_Rew)
        self.history['Avg_Rew'].append(avg_rew)
            
            
    def init_weights(self):
        
        self.critic_train(np.ones([1,self.num_states],dtype=np.float32),np.ones([1,self.num_actions],dtype=np.float32))
        self.critic_target(np.ones([1,self.num_states],dtype=np.float32),np.ones([1,self.num_actions],dtype=np.float32))
        
        self.actor_train(np.ones([1,self.num_states],dtype=np.float32))
        self.actor_target(np.ones([1,self.num_states],dtype=np.float32))
        
        critic_train_variables = np.array(self.critic_train.weights, dtype=object)
        actor_train_variables = np.array(self.actor_train.weights, dtype=object)
        
        critic_targ_variables = np.array(self.critic_target.weights, dtype=object)
        actor_targ_variables = np.array(self.actor_target.weights, dtype=object)
        
        for c_train, c_targ in zip(critic_train_variables,critic_targ_variables):
            c_targ.assign(c_train.numpy()) 
            
        for a_train, a_targ in zip(actor_train_variables,actor_targ_variables):
            a_targ.assign(a_train.numpy())
            
            
    def _copy_weights(self,soft_tau):
        critic_train_variables = np.array(self.critic_train.weights, dtype=object)
        actor_train_variables = np.array(self.actor_train.weights, dtype=object)
        
        critic_targ_variables = np.array(self.critic_target.weights, dtype=object)
        actor_targ_variables = np.array(self.actor_target.weights, dtype=object)
        
        #Soft update
        soft_critic_update= soft_tau*critic_train_variables + (1-soft_tau)*critic_targ_variables
        soft_actor_update= soft_tau*actor_train_variables + (1-soft_tau)*actor_targ_variables
        
        for c_soft, c_targ in zip(soft_critic_update,critic_targ_variables):
            c_targ.assign(c_soft.numpy()) 
            
        for a_soft, a_targ in zip(soft_actor_update,actor_targ_variables):
            a_targ.assign(a_soft.numpy())
            
        
    def train(self, t):
        
        
        if t >= self.train_after and t % self.train_every == 0:
            
            for j in range(self.gradient_steps):
            
                ids = np.random.randint(low=0,high=len(self.rep_buffer['s']), size=self.batch_size)
                states = np.asarray([self.rep_buffer['s'][i] for i in ids],dtype=np.float32)
                actions = np.asarray([self.rep_buffer['a'][i] for i in ids],dtype=np.float32)
                rewards = np.asarray([[self.rep_buffer['r'][i]] for i in ids],dtype=np.float32)
                next_states = np.asarray([self.rep_buffer['sn'][i] for i in ids],dtype=np.float32)
                dones = np.asarray([[self.rep_buffer['done'][i]] for i in ids])
            
                #Target Policy Smoothing
                noise_reg=np.clip(np.float32(np.random.normal(0,self.gauss_reg,[self.batch_size,self.num_actions])),
                                  -self.clip_reg,self.clip_reg) 
                action_target= np.clip(self.actor_target(next_states)+noise_reg,
                                       np.float32(self.action_low),np.float32(self.action_high))
            
                #Clipped Double-Q Learning
                Q1_target, Q2_target = self.critic_target(next_states,action_target)
                
                y=np.where(dones,rewards, 
                           rewards + self.gamma*np.minimum(Q1_target,Q2_target))
    
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    
                    Q1_train, Q2_train = self.critic_train(states,actions)
                    
                    critic1_loss = tf.math.reduce_sum(tf.math.square(y - Q1_train))
                    critic2_loss = tf.math.reduce_sum(tf.math.square(y - Q2_train))
                    #critic_loss = critic1_loss + critic2_loss
                    
                    critic_variables= self.critic_train.trainable_variables
                    
                    critic1_gradients= tape1.gradient(critic1_loss,critic_variables)
                    self.critic_optimizer.apply_gradients(zip(critic1_gradients,critic_variables))
                    
                    critic2_gradients= tape2.gradient(critic2_loss,critic_variables)
                    self.critic_optimizer.apply_gradients(zip(critic2_gradients,critic_variables))
                    
                
                #Delayed Policy Updates
                if j % self.delayed_train == 0:
                
                    with tf.GradientTape() as tape:
            
                        action_next=self.actor_train(states)
                        Q1_train_n, _ = self.critic_train(states,action_next)
                        
                        actor_loss= -tf.math.reduce_mean(Q1_train_n)
                
                        actor_variables= self.actor_train.trainable_variables
                        actor_gradients= tape.gradient(actor_loss,actor_variables)
                        self.actor_optimizer.apply_gradients(zip(actor_gradients,actor_variables))
                
                    self._copy_weights(self.soft_tau)
                    
                
        #return critic_loss, actor_loss
     
    
    def plot_results(self):
        plt.figure()
        plt.plot(self.history['Tot_Rew'],'o' ,markerfacecolor='white', label="Episode Reward")
        plt.plot(self.history['Avg_Rew'], label="Average Reward")
        plt.grid()
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.legend()
        
    def save_history(self, file_name):
        #File name should be "file_name.data"
        
        with open(file_name, 'wb') as filehandle:
   
            pickle.dump(self.history, filehandle)
            
            
    
def load_history(file_name):
    with open(file_name, 'rb') as filehandle:
    # read the data as binary data stream
        history = pickle.load(filehandle)
        
    plt.figure()
    plt.plot(history['Tot_Rew'],'o' ,markerfacecolor='white', label="Episode Reward")
    plt.plot(history['Avg_Rew'], label="Average Reward")
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    
    
    
    
    
    
    
    