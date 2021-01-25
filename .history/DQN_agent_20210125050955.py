from keras.layers import Input, Conv2D, Dense, concatenate
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense, DepthwiseConv2D,  Lambda, Add, Average,   Conv1D, Conv2D, Subtract, Activation, LocallyConnected1D, Reshape, concatenate, Concatenate, Flatten, Input, Dropout, MaxPooling1D,  MaxPooling2D
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
np.random.seed(0)
import keras

class DQN:

    """ Implementation of deep q learning algorithm """
    currEpisode = 0
    env = 0
    def __init__(self, action_space, state_space, model=None):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .9
        self.batch_size = 32
        self.epsilon_min = .01
        self.epsilon_decay = .0005
        self.burn_limit = .001
        self.learning_rate = .7e4
        memory_size = 20000
        self.modelname ='D3QNmodel'
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=5000)
        if model == None:
            self.model = self.build_modelGPU()
        else:
            self.model = model 

    def saveModel(self, score="n.a"):
        print("Model-F3DA-"+str(int(score))+"-" +
              str(DQN.currEpisode) + " saved! ")
        self.model.save(
            "savedModels\\CNN-{}-{:.2f}.h5".format(DQN.currEpisode, score), overwrite=True)

    def build_modelPar(self,input_shape=(4, 100, 100,)):

        self.network_size = 12*4 + 6*4 + 12

        digit_0 = Input(shape=(40000,))
        t = Reshape(input_shape)(digit_0)

        digit_a = Input(shape=input_shape)
        X = Conv2D(64, 4, strides=(2), activation="relu",  padding="valid",
                   kernel_initializer='he_uniform', data_format='channels_first')(t)
        X = Conv2D(64, 3, strides=(1), activation="relu",  padding="valid",
                   kernel_initializer='he_uniform', data_format='channels_first')(X)
        out_a = Flatten()(X)

        digit_b = Input(shape=input_shape)
        x = Flatten()(t)
        x = Dense(32, activation="relu")(x)
        out_b = Dense(32, activation="relu")(x)        

        concatenated = concatenate([out_a, out_b])
        # model_final.add(Reshape((4,11,2), input_shape=(88,)))
        # model_final.add(concatted)
        # model_final.add(Flatten())
        # model_final.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        # model_final.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        out_c = Dense(128, activation='relu',
                      kernel_initializer='he_uniform')(concatenated)
        
        out_c = Dense(64, activation='relu',
                      kernel_initializer='he_uniform')(out_c)

        state_value = Dense(1, kernel_initializer='he_uniform')(out_c)
        state_value = Lambda(lambda s: K.expand_dims(
            s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        action_advantage = Dense(
            self.action_space, activation='linear', kernel_initializer='he_uniform')(out_c)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
            a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage])

        model_final = Model([digit_0], out,name='ParallelCNNmodel')

        model_final.compile(loss="mse", optimizer=RMSprop(
            lr=self.learning_rate, rho=0.95, decay=0.0, epsilon=self.epsilon), metrics=["accuracy"])
        print(model_final.summary())
        return model_final

    def mean(input):
        return K.mean(input, axis=1)

    def build_modelGPU(self, input_shape=(4, 100, 100,), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(40000,))
        X = X_input

        X = Reshape(input_shape)(X)
        X = Conv2D(64, 4, strides=(2), activation="relu",  padding="valid",
                   kernel_initializer='he_uniform', data_format='channels_first')(X)
        X = Conv2D(64, 3, strides=(1), activation="relu",  padding="valid",
                   kernel_initializer='he_uniform', data_format='channels_first')(X)
        X = Flatten()(X)
        X = Dense(self.network_size,  activation="relu",
                  kernel_initializer='he_uniform')(X)
        X = Dense(64,  activation="relu", kernel_initializer='he_uniform')(X)
        state_value = Dense(1,kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(
            s[:, 0], -1), output_shape=(self.action_space,))(state_value)
        action_advantage = Dense(
            self.action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
            a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage]) 

        model = Model(inputs=X_input, outputs=out, name=self.modelname)
        model.compile(loss="mse", optimizer=RMSprop(
            lr=self.learning_rate, rho=0.95, epsilon=self.epsilon), metrics=["accuracy"])
        model.summary()
        return model

    #cpu - channels
    def build_modelCPU(self, input_shape=(100, 100, 4,), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(40000,))
        X = X_input

        X = Reshape(input_shape)(X)
        X = Conv2D(64, 5, strides=(3), activation="relu",
                   padding="valid", kernel_initializer='he_uniform')(X)
        X = Conv2D(64, 4, strides=(2), activation="relu",
                   padding="valid", kernel_initializer='he_uniform')(X)
        X = Conv2D(64, 3, strides=(1), activation="relu",
                   padding="valid", kernel_initializer='he_uniform')(X)
        X = Flatten()(X)
        X = Dense(self.network_size*2,  activation="relu",
                  kernel_initializer='he_uniform')(X)
        X = Dense(self.network_size,  activation="relu",
                  kernel_initializer='he_uniform')(X)
        X = Dense(64,  activation="relu", kernel_initializer='he_uniform')(X)
        state_value = Dense(1)(X)
        state_value = Lambda(lambda s: K.expand_dims(
            s[:, 0], -1), output_shape=(self.action_space,))(state_value)
        action_advantage = Dense(
            self.action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
            a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage])
        out = Dense(self.action_space, activation='linear')(out)
        model = Model(inputs=X_input, outputs=out, name=self.modelname)
        model.compile(loss="mse", optimizer=RMSprop(
            lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        logdir = pathlib.Path("logs")
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch], dtype=float)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch], dtype=float)
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        # states=np.reshape(np.squeeze(states),(100,4,11,2))
        # next_states=np.reshape(np.squeeze(next_states),(100,4,11,2))
        targets = (rewards*1) + self.gamma * \
            (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode,  graphics=True, ch=300,  lchk = 0 , model=None):
    env = StarShipGame(graphics)
    loss = [] 
    action_space = 6
    state_space = 40000
    max_steps = 98*9
    # agent =ic(DQN(action_space, state_space,  model=ic(keras.models.load_model('CNN-990--0.40.h5'))))
    # for e in range(990,episode):
    agent =ic(DQN(action_space, state_space,  model=model))
    for e in range(lchk,episode):   
        state = env.resetNew()
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
            if (env.save):
                agent.saveModel(score)
                env.save = False
            action = agent.act(state)
            reward, next_state, done = env.stepNew(action)
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
            agent.replay()
            if done:
                print(i)
                print("episode: {}/{}, score: {}, lr : {}".format(e,
                                                                  episode, score, agent.learning_rate))

                break
        loss.append(score)
        plot_loss(e, loss[1:])
            
          

        if DQN.currEpisode % ch == 0:
            agent.saveModel(score)

    agent.saveModel(score)
    return loss


def plot_loss(ep, loss):
    try: 
        plt.plot([i for i in range(ep)], loss)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.title('Episodal Loss')
        plt.savefig('logs\\loss_plot.png')          
    except: 
        print("loss_plot error. Skipping plotting.\n")
        pass   
    # #png 2  
    #     plt.clf()
    #     df = pd.read_csv('logs\\history.csv')
    #     x1 =df.epoch.values
    #     plt.title('Accuracy and Loss over epochs')
    #     plt.xlabel('epochs')
    #     plt.plot([i for i in range(len(x1))], df.accuracy.values,label='Accuracy')
    #     plt.plot([i for i in range(len(x1))], df.loss.values,label='Loss')
    #     plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    #     plt.savefig('logs\\Acc+lot_plot.png')
    #     plt.clf()
    # 
if __name__ == '__main__':
    ep = 3000
    loss = train_dqn(ep)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
