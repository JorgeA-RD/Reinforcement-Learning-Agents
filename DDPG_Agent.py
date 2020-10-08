import tensorflow as tf
import numpy as np

#policy network
class actor_netw(tf.keras.Model):
    
    def __init__(self,num_states, actor_hidden_units, num_actions,action_high):
        super(actor_netw,self).__init__()
        self.actor_input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.actor_hidden_layers = []
        self.action_high = action_high
        
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
        
        
        self.critic_input_layer = tf.keras.layers.InputLayer(input_shape=(num_states+num_actions,))
        self.critic_hidden_layers = []
        
        for i in critic_hidden_units:
            self.critic_hidden_layers.append(
                    tf.keras.layers.Dense(i, activation= tf.nn.leaky_relu, kernel_initializer= 'RandomNormal'))
                    
        self.critic_output_layer = tf.keras.layers.Dense(1, activation= 'linear', kernel_initializer= 'RandomNormal')

    @tf.function
    def call(self,input_state,input_action):
        
        inputs=tf.concat((input_state,input_action),1)
        z = self.critic_input_layer(inputs)
        for layer in self.critic_hidden_layers:
            z = layer(z)
        output = self.critic_output_layer(z)
        
        return output
    
class DDPG_Agent:
    
    def __init__(self,num_states,actor_hidden_units,critic_hidden_units,num_actions,
                 action_low,action_high,gamma,eta_critic,eta_actor,delta_huber,batch_size,cap_buffer):
        
        
        self.num_actions=num_actions
        self.action_low=action_low
        self.action_high=action_high
        
        self.gamma=gamma
        self.critic_optimizer = tf.optimizers.Adam(eta_critic)
        self.actor_optimizer = tf.optimizers.Adam(eta_actor)
        self.delta_huber = delta_huber
        self.batch_size=batch_size

        
        self.rep_buffer = {'s':[],'a':[],'r':[],'sn':[],'done':[]}
        self.cap_buffer = cap_buffer
        
        
        self.actor_train=actor_netw(num_states,actor_hidden_units,num_actions,action_high)
        self.actor_target=actor_netw(num_states,actor_hidden_units,num_actions,action_high)
        self.actor_target.trainable=False
        
        self.critic_train=critic_netw(num_states, critic_hidden_units, num_actions)
        self.critic_target=critic_netw(num_states, critic_hidden_units, num_actions)
        self.critic_target.trainable=False
    
    
    
    def action_selection(self,state,gauss_scale):
        #uncorrelated zero-mean Gaussian noise
        noise=np.random.normal(0,gauss_scale,self.num_actions)
        #noise_b=np.random.normal(0,gauss_scale/3,1)
        #noise=np.append(noise_a,noise_b)
        action=self.actor_train(np.asarray([state],dtype=np.float32))
        
        action=np.reshape(action,self.num_actions) + noise
        action=np.clip(action,self.action_low,self.action_high)

        return action.tolist()
    
        
    def fill_replay_buff(self,experience):
        if len(self.rep_buffer['s']) == self.cap_buffer:
            for key in self.rep_buffer.keys():
                self.rep_buffer[key].pop(0)
        for key, value in experience.items():
            self.rep_buffer[key].append(value)
            
    def train(self):
        
        if len(self.rep_buffer['s']) == self.cap_buffer:
            
            ids = np.random.randint(low=0,high=self.cap_buffer, size=self.batch_size)
            states = np.asarray([self.rep_buffer['s'][i] for i in ids],dtype=np.float32)
            actions = np.asarray([self.rep_buffer['a'][i] for i in ids],dtype=np.float32)
            rewards = np.asarray([self.rep_buffer['r'][i] for i in ids],dtype=np.float32)
            next_states = np.asarray([self.rep_buffer['sn'][i] for i in ids],dtype=np.float32)
            dones = np.asarray([self.rep_buffer['done'][i] for i in ids])
            
            
            action_target= self.actor_target(next_states)
            
            y=np.where(dones,rewards, rewards + self.gamma*self.critic_target(next_states,action_target))
    
            with tf.GradientTape() as tape:
                
                #standard loss
                critic_loss= tf.math.reduce_sum(tf.math.square(y - self.critic_train(states,actions)))
                
                #hubber loss
                #h_loss= tf.keras.losses.Huber(delta=self.delta_huber,reduction=tf.keras.losses.Reduction.SUM)
                #critic_loss=h_loss(y,self.critic_train(states,actions))
                
                critic_variables= self.critic_train.trainable_variables
                critic_gradients= tape.gradient(critic_loss,critic_variables)
                self.critic_optimizer.apply_gradients(zip(critic_gradients,critic_variables))
                
            with tf.GradientTape() as tape:
            
                action_next=self.actor_train(states)
                actor_loss= -tf.math.reduce_mean(self.critic_train(states,action_next))
                
                actor_variables= self.actor_train.trainable_variables
                actor_gradients= tape.gradient(actor_loss,actor_variables)
                self.actor_optimizer.apply_gradients(zip(actor_gradients,actor_variables))
                
            return critic_loss, actor_loss
        else:
            return 0, 0
       
        
    def copy_weights(self,soft_tau):
        critic_train_variables = np.array(self.critic_train.trainable_weights, dtype=object)
        actor_train_variables = np.array(self.actor_train.trainable_weights, dtype=object)
        
        critic_targ_variables = np.array(self.critic_target.non_trainable_weights, dtype=object)
        actor_targ_variables = np.array(self.actor_target.non_trainable_weights, dtype=object)
        
        #Soft update
        soft_critic_update= soft_tau*critic_train_variables + (1-soft_tau)*critic_targ_variables
        soft_actor_update= soft_tau*actor_train_variables + (1-soft_tau)*actor_targ_variables
        
        for c_soft, c_targ in zip(soft_critic_update,critic_targ_variables):
            c_targ.assign(c_soft.numpy()) 
            
        for a_soft, a_targ in zip(soft_actor_update,actor_targ_variables):
            a_targ.assign(a_soft.numpy())

    
    
    
    
    
    
    
    
    
    
    
    