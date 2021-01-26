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


env = StarShipGame()
class DQN:

    """ Implementation of deep q learning algorithm """
    currEpisode = 0
    env = 0
    def __init__(self, action_space, state_space, model=None):
        self.scores =0 
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = .1
        self.gamma = .97
        self.batch_size = 32
        self.epsilon_min = .01
        self.epsilon_decay = 1e-5
        self.burn_limit = .001
        self.learning_rate = .7e-4
        memory_size = 20000
        self.modelname ='D3QNmodel'
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=5000)
        if model == None:
            self.model = self.build_modelGPU()
            # self.target_model = self.build_modelGPU()
        else:
            self.model = model 
            # self.target_model =model 


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
    
    def logloss(y_true, y_pred):     #policy loss
	return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1) 
	# BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

    #loss function for critic output
    def sumofsquares(y_true, y_pred):        #critic loss
	return K.sum(K.square(y_pred - y_true), axis=-1)


    def build_modelGPU(self, input_shape=(4, 100, 100,1), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(40000,))
        X = X_input

        X = Reshape(input_shape)(X)        
        X = Conv2D(64, 5, strides=(3, 3),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_first")(X)
        X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_first")(X)
        X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_first")(X)
        X = Flatten()(X)
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
        X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
        
        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        if dueling:
            state_value = Dense(1, kernel_initializer='he_uniform')(X)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs = X_input, outputs = X)
        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
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


def train_dqn(episode,  graphics=True, ch=300,  lchk = 0 , model=None ,self=self):
    env = StarShipGame(graphics)
    self.env_name = "StarShip"
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
            next_state = env.getPixelsOnScreenNew()
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
                self.scores.append(score)
                print("Max: ",i," Ep: ",e)
                print("episode: {}/{}, score: {}, lr : {}".format(e,
                                                                  episode, score, agent.learning_rate))
                if e % self.REM_STEP == 0:
                        # self.update_target_model()
                       
                    # every episode, plot the result
                        average = self.PlotModel(i, e)

                break
        loss.append(score)
        plot_loss(e, loss[1:])
            
          

        if DQN.currEpisode % ch == 0:
            agent.saveModel(score)

    agent.saveModel(score)
    return loss

def test(self):
        self.load(self.Model_name)
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = env.step(action)
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

import pylab
pylab.figure(figsize=(18, 9))
def PlotModel(self, score, episode):

        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn: dqn = 'DDQN_'
        if self.Soft_Update: softupdate = '_soft'
        if self.dueling: dueling = '_Dueling'
        if self.epsilon_greedy: greedy = '_Greedy'
        if self.USE_PER: PER = '_PER'
        try:
            pylab.savefig(dqn+self.env_name+softupdate+dueling+greedy+PER+"_CNN.png")
        except OSError:
            pass

        return str(self.average[-1])[:5]
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