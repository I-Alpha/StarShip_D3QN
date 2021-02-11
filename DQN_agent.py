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
np.random.seed(0)
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
        self.gamma = .999
        self.batch_size =  64
        self.epsilon_min = .1
        self.epsilon_decay = 0.99997# 0.999998  (98 *4)
        # self.burn_limit = .001
        self.learning_rate = 0.0001
        self.replay_freq = 1
        self.memory = RingBuf(1000000)
        self.optimizer_model = 'Adam'
        self.log_data=[]
        self.log_history=[]
        if model == None:
            self.model =  build_LSTM(self)  # dfault _model
            # self.target_model = self.build_modelGPU()
        else:
            self.model = model
            # self.target_model =model
        self.modelname = self.model._name
        time_ = datetime.datetime.now
        self.savedir = "savedModels/"+self.model.name+"/"+time_().strftime("%m%d%h")+"/"

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 

    def replay(self):

        if self.memory.__len__() < self.batch_size:
            return

        minibatch = random.sample(
            self.memory.data[0:self.memory.__len__()], self.batch_size)
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
        history = self.model.fit(states, targets_full, verbose=0)
        self.log_history.append(history.history['loss'])
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    
gl_total_frames = 0
gl_score = 0
gl_loss = 0


def train_dqn(episode,  graphics=True, ch=300,  lchk=0, model=None, ):
    #loss = []
    action_space = 6
    state_space = env.REM_STEP*112
    DQN.REM_STEP = env.REM_STEP
    max_steps = 98*9
    agent = DQN(action_space, state_space,  model=model)
    agent.env_name = "StarShip"
    epsilon_log =[]
    
    for e in range(lchk, episode): 
      
        state = env.reset()
        ## Burnrate function
        # if agent.learning_rate < agent.burn_limit and DQN.currEpisode > 0:
        #     # after 1000 iterations learning rate will be 0.001
        #     agent.learning_rate += (.0000009)
        DQN.currEpisode += 1
        funcs = [lambda: (np.reshape(state, (1, state_space))),
                 lambda: (np.reshape(state, (1, len(state))))]
        # for func in funcs:
        #     try:
        #           state = func()
        #     except:
        #         pass
        state = funcs[0]()
        score = 0
        for i in range(max_steps):
            if i != 0:
                if i % 98 == 0:
                    env.time_multipliyer *= 1.5
                env.time_multipliyer += 0.01
                

           
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            next_state = env.getEnvStateOnScreen()  # do i need this?
            score += reward
            funcs = [lambda: (np.reshape(next_state, (1, state_space))), lambda: (
                np.reshape(next_state, (1, len(next_state))))]
            # for func in funcs:
            #     try:
            #         next_state = func()
            #     except:
            #          pass

            next_state = funcs[0]()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            for c in range(agent.replay_freq):
                agent.replay()

            # Add values to Tensorboard

            average = 0
            if done: 

                epsilon_log.append(agent.epsilon)
                agent.scores.append(score)
                agent.episodes.append(e)
                agent.average.append(sum(agent.scores) / len(agent.scores))
                agent.log_data.append(score)
                t1 =[ ]
                x1= []
                for i in agent.log_history:
                     t1.append(*i)
                     x1.append(sum(t1)/len(agent.log_history))
                PlotData("Episode_versus_score",["Episode","score" ],[agent.log_data,agent.average],["score","average"] )                      
                PlotData("Iteration_versus_loss",["Iteration","loss" ],[agent.log_history,x1],["loss","average"])
                t2 =[]
                x2= []
                for i in epsilon_log:
                     t2.append(i)
                PlotData("Iteration_versus_Epsilon",["Iteration","epsilon" ],[t2],["Epsilon"])      
                print("episode: {}/{}, score:  {:0.3f}, average: {}, epsilon: {}".format(e,
                                                                            episode, score,  str(agent.average[-1])[:5],agent.epsilon))

                # print("Max: ",i," Ep: ",e)
                # # print("episode: {}/{}, score: {}, lr : {}".format(e,global file_writer   t
                # training_summary = tf.Summary(value=[
                #     tf.Summary.Value(tag="loss", simple_value=gl_loss),
                #     tf.Summary.Value(tag="average", simple_value=float(average)),
                #     tf.Summary.Value(tag="score", simple_value=score),
                #     tf.Summary.Value(tag="max-step", simple_value=i),
                #     tf.Summary.Value(tag="dead obstacles", simple_value=env.obstacleGenerator.deadObstacles)
                #     ]) 
                if  env.save:
                    saveModel(agent,score) 
                    PlotData(agent.savedir+ "Episode_versus_score",["Episode","score" ],[agent.log_data,agent.average],["score","average"] )                      
                    PlotData(agent.savedir+"Iteration_versus_loss",["Iteration","loss" ],[agent.log_history,x1],["loss","average"])
                    PlotData(agent.savedir+"Iteration_versus_Epsilon",["Iteration","epsilon" ],[t2],["Epsilon"])      
                    env.save = False
                break
        #     else:
        #           training_summary = tf.Summary(value=[
        #             tf.Summary.Value(tag="loss", simple_value=gl_loss),
        #             tf.Summary.Value(tag="score", simple_value=gl_score),
        #             tf.Summary.Value(tag="max-step", simple_value=i),
        #             tf.Summary.Value(tag="dead obstacles", simple_value=env.obstacleGenerator.deadObstacles),
        #     ])
        # # file_writer.add_summary(training_summary, global_step=e)
        # global g
        # if done :
        #     file_writer.flush()
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
