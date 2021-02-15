from keras.callbacks import History
import time
import datetime
from tensorflow.keras import initializers
import keras
import itertools
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
import sys
from icecream import ic
from Models import *
from Utilities import *
np.random.seed(5)
env = StarShipGame(True)


log_data = []
 

class DQN:

    """ Implementation of deep q learning algorithm """
    currEpisode = 0 
    REM_STEP =0

    def __init__(self, action_space, state_space, model=None):
        self.env_name = 0
        self.scores, self.episodes, self.average = [], [], []
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .9
        self.tau =0.1
        self.batch_size =  4
        self.epsilon_min = .1
        self.epsilon_decay = 0.999# 0.999998  (98 *4)
        self.epsilon_decay_steps = 500000
        self.epsilon_log = []
        # self.burn_limit = .001
        self.learning_rate = 0.00025
        self.update_step =10000
        self.replay_init =500
        self.memory = RingBuf(500000)
        self.optimizer_model = 'Adam'
        self.log_data=[]
        self.log_history=[]
        self.epsilons = np.linspace(self.epsilon, self.epsilon_min, self.epsilon_decay_steps)# The epsilon decay schedule
 
        if model == None:
            self.model = self.build_model()  # dfault _model
            self.target_model = self.build_model()

        else:
            self.model = model
            self.target_model = odel
            # self.target_model =model
        self.modelname = self.model._name
        time_ = datetime.datetime.now
        self.savedir = "savedModels/"+self.model.name+"/"+time_().strftime("%m%d%h")+"/"
    
    def build_model(self):
              return (FCTime_distributed_model(self))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):          
        if self.epsilon > np.random.rand():
            # explore
            return np.random.choice(self.action_space)
        else:
            # exploit
            state = self._reshape_state_for_net(state)
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)
    def replay(self):         
        
        minibatch = random.sample(self.memory.data[0:len(self.memory)], self.batch_size)
        minibatch_new_q_values = []
        for experience in minibatch:
            state, action, reward, next_state, done = experience            
            state = self._reshape_state_for_net(state)
            experience_new_q_values = self.model.predict(state)[0]
            if done:
                q_update = reward
            else:
                
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                online_net_selected_action = np.argmax(self.model.predict(next_state))
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_model.predict(next_state)[0][online_net_selected_action]
                q_update = reward + self.gamma * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.model.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)
    
    def update_target_model(self):
        q_network_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_model_theta):
            target_weight = target_weight * (1-self.tau ) + q_weight * self.tau 
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)

    def _reshape_state_for_net(self, state):
        return np.reshape(state,(1, self.state_space))  

    
gl_total_frames = 0
gl_score = 0
gl_loss = 0


def train_dqn(episodes,  graphics=True, ch=300,  lchk=0, model=None, ):

    def saveResults(agent): 
                agent.epsilon_log.append(agent.epsilon)
                agent.scores.append(score)
                agent.episodes.append(e)
                agent.average.append(sum(agent.scores) / len(agent.scores))
                agent.log_data.append(score)

    def plotResults(agent):
                t1 =[ ]
                x1= []
                for i in agent.log_history:
                        i = i[0]
                        t1.append(i*-1)
                        x1.append(sum(t1)/len(agent.log_history))
                PlotData("Episode_versus_score",["Episode","score" ],[agent.log_data,agent.average],["score","average"] )                      
                PlotData("Iteration_versus_loss",["Iteration","loss" ],[t1,x1],["loss","average"])
                t2 =[]
                x2= []
                for i in agent.epsilon_log:
                        t2.append(i)
                PlotData("Iteration_versus_Epsilon",["Iteration","epsilon" ],[t2],["Epsilon"])      
                print("episode: {}/{}, score:  {:0.3f}, average: {}, epsilon: {}".format(e,
                                                                            episodes, score,  str(agent.average[-1])[:5],agent.epsilon))
                sys.stdout.flush()

    #loss = []
    action_space = 6
    state_space = env.REM_STEP*54

    DQN.REM_STEP = env.REM_STEP 
    max_steps = 98*9
    agent = DQN(action_space, state_space,  model=model)
    agent.env_name = "StarShip"     
    total_t = 0
    
    for e in range(lchk, episodes): 

        state = (env.reset())
        ## Burnrate function
        # if agent.learning_rate < agent.burn_limit and DQN.currEpisode > 0:
        #     # after 1000 iterations learning rate will be 0.001
        #     agent.learning_rate += (.0000009)
        DQN.currEpisode = e
        
        #           state = func()
        #     except:
        #         pass
        score = 0
        for i in itertools.count(): 
            if i != 0:
                if i % 98 == 0:
                    env.time_multipliyer *= 1.5
                env.time_multipliyer += 0.01              
           
            action = (agent.act(state))
            reward, next_state, done = env.step(action) 
            score += reward
          
            (agent.remember(state, action, reward, next_state, done))      
            if  (agent.replay_init < len(agent.memory))  :
                    agent.replay()
                    agent.epsilon = agent.epsilons[min(total_t,agent.epsilon_decay_steps - 1)]
                    if(total_t % agent.update_step== 0):
                        agent.update_target_model()
            #append to lists 
            if done: 
                state = env.reset()
                saveResults(agent) 
                plotResults(agent)               
                if  env.save:
                    saveModel(agent,score) 
                    env.save = False
                if DQN.currEpisode % ch == 0:
                    saveModel(agent,score) 
                break
            else:
                state = next_state     
        
            total_t += 1       
                
            

     


    def test(self):
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
