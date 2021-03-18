from keras.callbacks import History
import time
import datetime
from tensorflow.keras import initializers
import keras
import pylab
from tensorflow import keras
from keras.layers import Input, Conv2D, Dense, concatenate
from memory_profiler import profile
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense,  LeakyReLU, DepthwiseConv2D,  Lambda,  Add, Average, LSTM, TimeDistributed, Conv1D, Conv2D, Subtract, Activation, LocallyConnected2D, LocallyConnected1D, Reshape, concatenate, Concatenate, Flatten, Input, Dropout, MaxPooling1D,  MaxPooling2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from StarShip import StarShipGame
from keras.models import Model
from keras.models import Model
from keras.layers import LSTM, Input, concatenate
from keras.optimizers import Adagrad, RMSprop
from keras.metrics import Mean
from keras import backend as K
from PER import *
import pathlib
import tensorflow as tf
import pandas as pd
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as FF
from icecream import ic
from Models import *
from Utilities import *
import itertools
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(5)
env = StarShipGame(True)


log_data = []
 


class DQN:

    """ Implementation of deep q learning algorithm """
    currEpisode = 0 
    REM_STEP =0
    startCheckPoint = 0
    def __init__(self, action_space, state_space, model=None):
        self.env_name = 0
        self.scores, self.episodes, self.average = [], [], []
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon =1
        self.epsilon_min=.1
        self.gamma = .9994
        self.batch_size = 64
        self.epsilon_decay = 0.998# 0.999998  (98 *4)
        self.epsilon_decay_episodes =1000
        self.epsilon_log = []
        # self.burn_limit = .001
        self.learning_rate = 0.001 
        self.replay_freq = 1
        self.start_episode =0
        self.tau =.1
        self.update_step =4
        self.target_update_step = 250       
        self.t_count =0
        self.memory = Memory(1000000)
        self.optimizer_model = 'RMSProp'#not in use
        self.log_data=[]
        self.log_history=[]
        self.epsilons = np.linspace(self.epsilon, self.epsilon_min, self.epsilon_decay_episodes)# The epsilon decay schedule
        if model == None:
            self.model = self.build_model()  # dfault _model
            self.target_model = self.build_model()
        else:
            self.model = model
            self.target_model = model
        self.target_model.set_weights(self.model.get_weights())
        self.modelname = self.model._name
        self.target_modelname = self.model._name+"_target"
        time_ = datetime.datetime.now

        self.savedir = "savedModels/"+self.model.name+"/"+time_().strftime("%m%d%h")+"/"
    
    def build_model(self):
              return build_1CNNBase(self)

    def memorize(self, state, action, reward, next_state, done):
        # Calculate TD-Error for Prioritized Experience Replay
            # td_error = reward + self.gamma * np.argmax(self.model.predict(next_state)[0]) - np.argmax(
            #     self.target.predict(state)[0])
            # # Save TD-Error into Memory
            # self.memory.add(td_error, (state, action, reward, next_state, done))
            target = self.model.predict(state,batch_size=1)
            old_val = target[0][action]
            target_val = self.target_model.predict(next_state,batch_size=1).data
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.argmax(target_val)

            error = abs(old_val - target[0][action])

            self.memory.add(error, (state, action, reward, next_state, done))

    def act(self, state):
            if np.random.rand() <= self.epsilon:  # Exploration
            #    if DQN.currEpisode <=  self.startEpisode:
                #     return (random.choices(population=range(6),weights=(0.32,0.32,0.05,0.15,0.1,0.05),
                # k=1)).pop()   # weighted exploration 
                return random.randrange(self.action_space)            
            act_values = self.model.predict(state, batch_size=1)
            return np.argmax(act_values[0])  # returns action (Exploitation)
    temp= 0
    def decrement_epsilon(self):
        if self.t_count==0:
            temp = total_t
        if self.epsilon > self.epsilon_min:
            # if self.currEpisode  > self.epsilon_decay_episodes/2:
                try : 
                    if DQN.currEpisode >= self.start_episode: # Epsilon Update
                    #  self.epsilon *= self.epsilon_decay
                       self.epsilon = self.epsilons[DQN.currEpisode- self.start_episode-0]
                except:
                       self.epsilon *= self.epsilon_decay
                       pass
            # else:
            #       self.epsilon *= self.epsilon_decay
                finally:
                    return 
        self.epsilon = 0.1

    def update_target_model(self):
        q_network_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_model_theta):
            target_weight = target_weight * (1-self.tau ) + q_weight * self.tau 
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)

    def reset_target_network(self):
        """
        Updates the target DQN with the current weights of the main DQN.
        """ 
        self.target_model.set_weights(self.model.get_weights())


    def replay(self):
            
            global data_history
        # Train the model for one epoch 
            # batch, idxs, is_weight = (self.memory.sample(self.batch_size))
            # for i in range(self.batch_size):
            #     state, action, reward, next_state, done = batch[i]
            #     if not done:
            #         target = (reward + self.gamma * np.amax(self.model.predict_on_batch(next_state)[0]))
            #     else:
            #         target = reward
            #     target_f = self.model.predict_on_batch(state)
            #     target_f[0][action] = target
            #     # Gradient Update. Pay attention at the sample weight as proposed by the PER Paper               
            #     history = self.model.fit(state, target_f, epochs=1, verbose=0, sample_weight=np.array([is_weight[i]]))
            # self.log_history.append(history.history["loss"])

            minibatch, idxs, is_weight =self.memory.sample(self.batch_size)
            states = np.array([i[0] for i in minibatch], dtype=float)
            actions = np.array([i[1] for i in minibatch])
            rewards = np.array([i[2] for i in minibatch])
            next_states = np.array([i[3] for i in minibatch], dtype=float)
            dones = np.array([i[4] for i in minibatch])
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            targets = (rewards*1) + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
            targets_full = self.model.predict_on_batch(states)
            ind = np.array([i for i in range(self.batch_size)])
            targets_full[[ind], [actions]] = targets
            history = self.model.fit(states, targets_full, verbose=0,sample_weight=is_weight) 
            temp = {'loss':history.history['loss'][0], 'accuracy': history.history['accuracy'][0], 'mean_absolute_error': history.history['mean_absolute_error'][0]}                   
            data_history=data_history.append(temp,ignore_index=True,sort=False)          
           
            self.t_count +=1  

total_t =0
score = 0
columnslist = ['score','average_score','epsilon','episode','loss','accuracy']
data =  pd.DataFrame(columns= ['score','average_score', 'epsilon' ])    
data_history =  pd.DataFrame(columns=['loss','accuracy','mean_absolute_error'])    
 

import _thread
def train_dqn(episode,  graphics=True, ch=300,  lchk=0, model=None, ):

    def saveResults(agent,e): 
                global data  
                data.loc[total_t, 'epsilon'] = agent.epsilon
                data.loc[total_t,'score'] = score
                data.loc[total_t,'average_score'] =  data['score'].mean()

    
    def plotResults(s=False,p=True):     
                global data , fig, g_plt, axes, data_history 
                t=0 
                fig,axes=plt.subplots(3,2)
                axes=np.reshape(axes,(-1))
                fig.set_size_inches(15,10)   
                t1=0
                for i in data.columns:
                    axes[t1].set_title(i)  
                    t1+=1
                for i in data.columns:
                    data[i].plot( ax=axes[t])                     
                    t+=1               
                if len(data_history.index.values) > 0:
                    for x in data_history.columns:
                        axes[t1].set_title(x) 
                        t1+=1  
                    for x in data_history.columns:
                        data_history[x].plot( ax=axes[t]) 
                        t+=1
                if s:
                    plt.savefig(agent.savedir +".png");
                if p:
                    plt.savefig(agent.model.name + ".png");  

                plt.close()
                 

    global data, data_history    

    action_space = 6
    state_space =env.COLS * env.REM_STEP 
    DQN.REM_STEP = env.REM_STEP 
    agent = DQN(action_space, state_space,  model=model)
    agent.env_name = "StarShip" 
    global score
    global total_t 
    for e in range(lchk, episode):            
  
        score = 0
        DQN.currEpisode = e
        state = env.reset()
        funcs = [lambda: (np.reshape(state, (1, state_space ))),
                 lambda: (np.reshape(state, (1, len(state))))]
        state = funcs[0]()      

        for i in itertools.count():
            
            if i != 0:
                if i % 98 == 0:
                    env.time_multipliyer *= 1.5
                env.time_multipliyer += 0.01              
           
            action = agent.act(state)
            reward, next_state, done = env.step(action) 
            funcs = [lambda: (np.reshape(next_state, (1, state_space ))), lambda: (
                np.reshape(next_state, (1, len(next_state))))]
            next_state = funcs[0]() 
            agent.memorize(state, action, reward, next_state, done)
            state = next_state      
            score += reward
          
            #appe6nd to lists 
            if total_t%agent.update_step== 0 and e - lchk >= agent.start_episode:
                    agent.replay()                      
                    if agent.t_count%agent.target_update_step ==0 and agent.t_count >= agent.target_update_step:
                        agent.reset_target_network()
                        print("Target Network updated!")         
            if done:               
                    saveResults(agent,e)
                    if env.save:
                        saveModel(obj=agent,data=data,score=score,checkpoint= lchk)      
                        plotResults(True)   
                        env.save = False  
                    if env.plot:
                        plotResults()
                        env.plot = False                    
                    print("episode: {}/{}, score:  {:.3f}, average: {:.3f}, epsilon: {:.3f} total_t: {}".format(e,
                                                                                episode, data['score'].values[-1],  data['average_score'].values[-1],agent.epsilon,total_t))
        
                    break          
            total_t+=1
            agent.decrement_epsilon()                    
                   
        if DQN.currEpisode % ch == 0:
                saveModel(obj=agent,data=data,score=score,checkpoint= 0) 
                plotResults(True) 
            


     


    def test(episode,  graphics=True, ch=300,  lchk=0, model=None,):       
        for e in range(self.EPISODES):
            state = env.reset()
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = env.step(action)
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES))
                    break




if __name__ == '__main__':
    ep = 3000
    train_dqn(ep)
