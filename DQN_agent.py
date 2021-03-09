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
        self.epsilon =1
        self.epsilon_min=.1
        self.gamma = .999
        self.batch_size = 128
        self.epsilon_decay = 0.99995# 0.999998  (98 *4)
        self.epsilon_decay_episodes = 10000
        self.epsilon_log = []
        # self.burn_limit = .001
        self.learning_rate = 0.0004 
        self.replay_freq = 1
        self.startEpisode =2
        self.update_ep =5
        self.memory = Memory(1000000)
        self.optimizer_model = 'RMSProp'
        self.log_data=[]
        self.log_history=[]
        self.epsilons = np.linspace(self.epsilon, self.epsilon_min, self.epsilon_decay_episodes)# The epsilon decay schedule
        if model == None:
            self.model = self.build_model()  # dfault _model
            # self.target_model = self.build_modelGPU()

        else:
            self.model = model
            # self.target_model =model
        self.modelname = self.model._name
        time_ = datetime.datetime.now
        self.savedir = "savedModels/"+self.model.name+"/"+time_().strftime("%m%d%h")+"/"
    
    def build_model(self):
              return FCTime_distributed_model(self)

    def memorize(self, state, action, reward, next_state, done):
        # Calculate TD-Error for Prioritized Experience Replay
            td_error = reward + self.gamma * np.argmax(self.model.predict(next_state)[0]) - np.argmax(
                self.model.predict(state)[0])
            # Save TD-Error into Memory
            self.memory.add(td_error, (state, action, reward, next_state, done))

    def act(self, state):
            if np.random.rand() <= self.epsilon:  # Exploration
            #    if DQN.currEpisode <=  self.startEpisode:
                #     return (random.choices(population=range(6),weights=(0.32,0.32,0.05,0.15,0.1,0.05),
                # k=1)).pop()   # weighted exploration 
                return random.randrange(self.action_space)            
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action (Exploitation)

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            try : 
                if DQN.currEpisode >= self.startEpisode: # Epsilon Update
                   self.epsilon *= self.epsilon_decay
            except:
                self.epsilon = 0.1 
                pass
        else:
            self.epsilon = 0.1

    def replay(self):
            if self.memory.tree.n_entries < self.batch_size:
                return
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
            targets = (rewards*1) + self.gamma * \
                (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
            targets_full = self.model.predict_on_batch(states)
            ind = np.array([i for i in range(self.batch_size)])
            targets_full[[ind], [actions]] = targets
            history = self.model.fit(states, targets_full, verbose=0, sample_weight = is_weight)
            self.log_history.append(history.history['loss'])
            self.decrement_epsilon()
                 
total_t =0
def train_dqn(episode,  graphics=True, ch=300,  lchk=0, model=None, ):

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
                PlotData("Episode_versus_Epsilon",["episode","epsilon" ],[t2],["Epsilon"]) 

                
    #loss = []
    action_space = 6
    state_space =env.COLS * env.REM_STEP 
    DQN.REM_STEP = env.REM_STEP 
    agent = DQN(action_space, state_space,  model=model)
    agent.env_name = "StarShip"
 
    
    for e in range(lchk, episode):       
      
        state = env.reset()
        DQN.currEpisode = e
        funcs = [lambda: (np.reshape(state, (1, state_space ))),
                 lambda: (np.reshape(state, (1, len(state))))]
        state = funcs[0]()
        score = 0

        for i in itertools.count():
        
            if i != 0:
                if i % 98 == 0:
                    env.time_multipliyer *= 1.5
                env.time_multipliyer += 0.01              
           
            action = agent.act(state)
            reward, next_state, done = env.step(action) 
            score += reward
            funcs = [lambda: (np.reshape(next_state, (1, state_space ))), lambda: (
                np.reshape(next_state, (1, len(next_state))))]
            next_state = funcs[0]()
            agent.memorize(state, action, reward, next_state, done)
            state = next_state        
            global total_t 
            if total_t % agent.update_ep==0:# and e>0 :
                agent.replay()
            average = 0
              
            #append to lists 
            if done:     
                
                if  env.save:
                    saveModel(agent,score) 
                    env.save = False 
                if e %5==0:
                    saveResults(agent) 
                    plotResults(agent) 
                print("episode: {}/{}, score:  {:0.3f}, average: {}, epsilon: {} total_t: {}".format(e,
                                                                            episode, score,  str(agent.average[-1])[:5],agent.epsilon,total_t))
                break
            
            total_t+=1

            # # if e == 0:           
            # if e == agent.startEpisode:   
            #     agent.replay() 
                   
        if DQN.currEpisode % ch == 0:
             saveModel(agent,score) 


     


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
