import tensorflow as tf
import numpy as np
import itertools as itt


class netw(tf.keras.Model):
   
    def __init__(self,num_states, hidden_units, num_actions):
        super(netw,self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states+num_actions,))
        self.hidden_layers = []
        
        for i in hidden_units:
            self.hidden_layers.append(
                    tf.keras.layers.Dense(i, activation= tf.nn.leaky_relu, kernel_initializer= 'RandomNormal'))
                    
        self.output_layer = tf.keras.layers.Dense(1, activation= 'linear', kernel_initializer= 'RandomNormal')

    @tf.function
    def call(self,input_state,input_action):
        
        inputs=tf.concat((input_state,input_action),1)
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        
        return output
    
def _action_space_gen(action_space_dim):
    
    if len(action_space_dim) == 1:
        action_space_list = [[i] for i in range(action_space_dim[0])]
        
    else:
        action_space_dummy = []

        for i in action_space_dim:
            a = list(range(i))
            action_space_dummy.append(a)
    
        action_space_list = list(itt.product(*action_space_dummy))
        
    return action_space_list, len(action_space_list)
    

class DDQN_Agent:
    
    def __init__(self,num_states, hidden_units, num_actions, action_space_dim, 
                 batch_size,gamma,cap_buffer,eta,delta_huber):
        
        self.num_actions=num_actions
        #self.action_space_dim = action_space_dim
        self.action_space_list, self.size_action_space = _action_space_gen(action_space_dim)

        self.gamma = gamma
        self.optimizer = tf.optimizers.Adam(eta)
        self.delta_huber = delta_huber
        
        self.rep_buffer = {'s':[],'a':[],'r':[],'sn':[],'done':[]}
        self.cap_buffer = cap_buffer
        self.batch_size= batch_size
        
        self.Q_train = netw(num_states, hidden_units, self.num_actions)
        self.Q_target = netw(num_states, hidden_units, self.num_actions)
        self.Q_target.trainable=False
        
    
    
    
    def action_selection(self, state, epsilon):
        #epsilon greedy action selection
        #n_a = len(action_space_list)
        
        if epsilon > np.random.random():
            id_action = np.random.randint(0,self.size_action_space)
            
        else:
            id_action = np.argmax(self.Q_train(np.tile(np.float32(state),[self.size_action_space,1]),
                                            np.asarray(self.action_space_list, dtype = np.float32)))
            
        return self.action_space_list[id_action]
    

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

    
    def train(self):
        if len(self.rep_buffer['s']) == self.cap_buffer:
            
            ids = np.random.randint(low=0,high=self.cap_buffer, size=self.batch_size)
            states = np.asarray([self.rep_buffer['s'][i] for i in ids],dtype=np.float32)
            actions = np.asarray([self.rep_buffer['a'][i] for i in ids],dtype=np.float32)
            rewards = np.asarray([[self.rep_buffer['r'][i]] for i in ids],dtype=np.float32)
            next_states = np.asarray([self.rep_buffer['sn'][i] for i in ids],dtype=np.float32)
            dones = np.asarray([[self.rep_buffer['done'][i]] for i in ids])
            
            #Standar DQN
            #targetnet_max = np.max(self.TargetNet(next_states),axis=2)
            #y = np.where(dones,rewards, rewards + self.gamma*targetnet_max )
            
            #Double DQN
            #targetnet_max = tf.math.reduce_sum(self.TargetNet(next_states)*
                                               #tf.one_hot(np.argmax(self.TrainNet(next_states)),self.num_actions),axis=2)
            
            Q_train_values = self.Q_train(np.repeat(next_states,self.size_action_space,axis = 0),
                                   np.float32(np.tile(self.action_space_list,[self.batch_size,1])))
            
            action_args = np.argmax(np.reshape(Q_train_values,[self.batch_size,self.size_action_space]),axis = 1)
            actions_max = np.asarray([self.action_space_list[i] for i in action_args],dtype=np.float32)
            
            y = np.where(dones,rewards, rewards + self.gamma*self.Q_target(next_states,actions_max))
            
            with tf.GradientTape() as tape:
                q_train = self.Q_train(states,actions)
                
                # standar norm
                loss= tf.math.reduce_mean(tf.square(y-q_train))
                
                #hubber loss
           #     h_loss= tf.keras.losses.Huber(delta=self.delta_huber,reduction=tf.keras.losses.Reduction.SUM)
           #     loss=h_loss(y,q_train)
                
                variables= self.Q_train.trainable_variables
                gradients= tape.gradient(loss,variables)
                self.optimizer.apply_gradients(zip(gradients,variables))
            return loss.numpy()        
        else:
            return 0
            #print('Replay buffer ',len(self.rep_buffer['s']),' of ', self.cap_buffer)
        
        
        
    
    def copy_weights(self,soft_tau):
            
        train_variables = np.array(self.Q_train.trainable_weights, dtype=object)
        targ_variables = np.array(self.Q_target.non_trainable_weights, dtype=object)
        
        #Soft update
        soft_update= soft_tau*train_variables + (1-soft_tau)*targ_variables

        
        for c_soft, c_targ in zip(soft_update,targ_variables):
            c_targ.assign(c_soft.numpy()) 
            
    def copy_weights_normal(self):
        train_variables = np.array(self.Q_train.trainable_weights, dtype=object)
        targ_variables = np.array(self.Q_target.non_trainable_weights, dtype=object)
        
        for train, targ in zip(train_variables,targ_variables):
            targ.assign(train.numpy())

                
            
            
            
            
        
        