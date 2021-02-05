from tensorflow import keras
from keras.layers import Input, Conv2D, Dense, concatenate
from memory_profiler import profile
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense, DepthwiseConv2D,  Lambda,  Add, Average, LSTM, TimeDistributed, Conv1D, Conv2D, Subtract, Activation, LocallyConnected2D, LocallyConnected1D, Reshape, concatenate, Concatenate, Flatten, Input, Dropout, MaxPooling1D,  MaxPooling2D
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
import pylab
import keras
from tensorflow.keras import initializers
import datetime
import time
from keras.callbacks import History 
from ring_buf import RingBuf
env = StarShipGame(True)
 
HUBER_LOSS_DELTA = 1.0
def huber_loss(y_true, y_predict):
        err = y_true - y_predict

        cond = K.abs(err) < HUBER_LOSS_DELTA
        L2 = 0.5 * K.square(err)
        L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
        loss = tf.where(cond, L2, L1)

        return K.mean(loss)
class DQN:

    """ Implementation of deep q learning algorithm """
    currEpisode = 0
    env = 0
    def __init__(self, action_space, state_space, model=None):
        self.env_name=0
        self.scores, self.episodes, self.average = [], [], []
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 99
        self.epsilon_min = .1
        self.epsilon_decay = 0.0000127551
        self.burn_limit = .001
        self.learning_rate = 25e-5
        self.memory = RingBuf(1000000)
        self.optimizer_model = 'RMSProp'

        if model == None:
            self.model = self.build_LocallyConnected1D() #dfault _model
            # self.target_model = self.build_modelGPU()
        else:
            self.model = model 
            # self.target_model =model 
        self.modelname = self.model._name


    def saveModel(self, score="n.a"):
        if DQN.currEpisode < 5: 
           return
        time_=datetime.datetime.now
        time_h =time_().strftime("%h")
        savedir = "savedModels/"+self.model.name+"/"+time_().strftime("%m%d")+"/"
        print("saving " + self.model.name + "-" + str(DQN.currEpisode)+str(int(score)) + "...." )
        name=self.model.name+time_().strftime("%h")+"_{}_{:0.2f}.h5".format(DQN.currEpisode, self.average[-1])
        try:            
            self.model.save( savedir+name, overwrite=True)                            
            self.saveLog(name+".txt",savedir) 
            print(name+ " saved! ")    
        except:
            print(name + " not saved! ")  
        try:
            pylab.savefig( savedir+ name)    
        except: 
            pass

    def saveLog(self,name="lastRun.txt", dir ="logs/fit/"):
       if DQN.currEpisode <5: 
           return
       try:
          f = open(dir+name, "w")          
          for i,q in log_data:
              f.write("{},{}\n".format(i,q))
          f.close()          
       except : 
          print("save log failed")

       try:
            pylab.savefig(dir+name+".png")
       except OSError:
            pass

        
    def build_modelPar(self,dueling = True,input_shape=(4,1,138)):
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init ="he_uniform" 
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)
        if dueling:
            x = Input(shape=(self.state_space,))
            t = Reshape(input_shape)(x)
            # a series of fully connected layer for estimating V(s) 
            y11= Dense(128, activation='relu',kernel_initializer=truncatedn_init, bias_initializer=const_init, use_bias=True)(t)
            y12 = Dense(128, activation='relu',kernel_initializer=truncatedn_init, bias_initializer=const_init, use_bias=True)(y11)  
            y13=  Flatten()(y12)   
            y14 = Dense(self.action_space, activation="linear",kernel_initializer=x_init)(y13)

            # a series of fully connected layer for estimating A(s,a)
           
           
            y20 = Flatten()(x)
            y21 = Dense(256, activation='relu',kernel_initializer=truncatedn_init, bias_initializer=const_init, use_bias=True)(y20)
            y22 = Dense(128, activation='relu',kernel_initializer=truncatedn_init, bias_initializer=const_init, use_bias=True)(y21)
            y23 = Dense(1, activation="linear",kernel_initializer=x_init)(y22)
            
            # a series of fully connected layer for estimating B(s,a)

            y30=  TimeDistributed(Dense(64,activation='relu'))(t)
            y31=  TimeDistributed(Dense(64,activation='relu'))(y30)
            y32=  Dense(1,activation='softmax')(y31)
            y33= Flatten()(y32)
            y34=  Dense(256,activation='relu')(y33) 
            y35=  Dense(64,activation='relu')(y34)            
            y36= Dense(self.action_space, activation='linear')(y35) 

            w = Concatenate(axis=-1)([y23,y14])            


            # combine V(s) and A(s,a) to get Q(s,a)
            z = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                       output_shape=(self.action_space))(w)
        else:
            x = Input(shape=(self.state_space,))

            # a series of fully connected layer for estimating Q(s,a)
            y1 = Dense(64, activation='relu')(x)
            y2 = Dense(64, activation='relu')(y1)
            z = Dense(self.action_space, activation="linear")(y2)

        model = Model(inputs=x, outputs=z)

        if self.optimizer_model == 'Adam':
            optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.summary()
        return model

    def build_modelPar_2(self,input_shape=(4,1,138,)):


        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)        
        truncatedn_init2 = initializers.TruncatedNormal(0, 2e-2)
        x_init ="he_uniform" 
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)
        const_init2=initializers.constant(2e-2)
        digit_0 = Input(shape=(4*138,))
        t = Reshape(input_shape)(digit_0)
        
        x = Dense(64, activation='relu',
                    kernel_initializer=x_init)(t)
        # x =    Dense(128, activation='relu',
        #             kernel_initializer=const_init,use_bias=True, bias_initializer=truncatedn_init)(x)     
        # x =    Dropout(0.4)(x)
        # x =    Reshape([1,-1,4])(x)
        # x =    MaxPooling2D((1,2),(2))(x) 
        # x=     Reshape([4,-1])(x)
        x = Dense(64, activation='softmax',
                    kernel_initializer=x_init)(x)
        out_a= (x)

        x = Dense(64, activation='relu',
                    kernel_initializer=x_init)(t)
        x = Dense(64, activation='softmax',
                    kernel_initializer='he_uniform')(x)   
        out_b= (x)
     
        x = Dense(64, activation='relu',
                    kernel_initializer=y_init)(t)        
        x = Dense(64, activation='softmax',
                    kernel_initializer='he_uniform')(x)
        out_c= (x)
      
        # x = Conv2D(4,(1,2),strides=(1,1), padding = "valid", activation="softmax", kernel_initializer=x_init , data_format="channels_first")(t) 
        # x = MaxPooling2D((4,2))(x)
        # x = LocallyConnected2D(8 ,(2,2),strides=(1,1), padding = "valid", activation="relu", kernel_initializer=x_init , data_format="channels_first")(x)
        # # x = MaxPooling2D((1,2))(x)
        # # x=Dropout(0.3)(x)
        # # x =Flatten()(x)
        # # x = Dense(64, activation="relu")(x)
        # # # x = Dense(64, activation="relu")(x)      
        # # out_c =  Dense(64, activation='relu',
        # #               kernel_initializer='he_uniform')(x)
        # out_c= (x)

        concatenated = concatenate([out_a,out_b,out_c])
        # model_final.add(Reshape((4,11,2), input_shape=(88,)))
        # model_final.add(concatted)
        # model_final.add(Flatten())
        # model_final.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        # # model_final.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        # out_d=  Dropout(0.4)(concatenated)
        # out_d=  MaxPooling2D((8,1))(out_d)
    
        out_d = Flatten()(concatenated)
        
        out_d = Dense(512, activation='relu',
                      kernel_initializer='he_uniform')(out_d) 

        # out_d = Dense(64, activation='relu',
        #               kernel_initializer='he_uniform')(out_d)            
         
        state_value = Dense(1, activation='softmax',kernel_initializer='he_uniform')(out_d)
        state_value = Lambda(lambda s: K.expand_dims(
            s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        action_advantage = Dense(
            self.action_space, activation='linear', kernel_initializer='he_uniform')(out_d)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
            a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage])

        model_final = Model([digit_0], out,name='ParallelCNNmodel')

        model_final.compile(loss="mean_squared_error",optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        #  RMSprop(lr=self.learning_rate, rho=0.95, decay=25e-5, moepsilon=self.epsilon), metrics=["accuracy"])
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

    
    
    def build_model_lstm(self, input_shape=(336,4), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(4*336,))
        X = X_input
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init ="he_uniform" 
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)
        X = Reshape(input_shape)(X)  
        # X =Conv1D(4,(4),(4),activation="relu",) (X)             
        # X = Conv2D(1, (1,4), strides=(1,1),padding="same",activation="relu", kernel_initializer=x_init,   data_format="channels_first")(X)
        X = LSTM(4, return_sequences=True, return_state=True)(X)
        X = Flatten()(X)
        X = Dense(512, activation="relu", kernel_initializer=x_init)(X)            
        X =Dense(256, activation="relu", kernel_initializer=x_init)(X)
        # X = Conv2D(64, 4, strides=(2),padding="valid",activation="elu", kernel_initializer=x_init,   data_format="channels_first")(X)
        # X = Conv2D(128, 4, strides=(2),padding="valid",activation="elu",kernel_initializer=x_init,   data_format="channels_first")(X) 3cnn
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes         
        # X = Dense(512, activation="relu", kernel_initializer=x_init)(X) 
        # X = Dense(64, activation="relu", kernel_initializer=x_init)(X)                
        X = Dense(256, activation="relu", kernel_initializer=x_init)(X)                  
        X = Dense(64, activation="relu", kernel_initializer=x_init)(X)              
        # # Hidden layer with 256 nodes
        # X = Dense(256, activation="relu", kernel_initializer=truncatedn_init, bias_initializer=const_init)(X)
        
        # # Hidden layer with 64 nodes
        # X = Dense(64, activation="relu", kernel_initializer=truncatedn_init, bias_initializer=const_init)(X)
 
        if dueling:
            state_value = Dense(1,kernel_initializer=x_init)(X)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(action_space,kernel_initializer=x_init)(X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",kernel_initializer='he_uniform', bias_initializer=const_init)(X)

        model = Model(inputs = X_input, outputs = X, name = '3CNN_model')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate),  metrics=["accuracy"])
        
        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model
 
    def build_LocallyConnected1D(self, input_shape=(138,4), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(4*138,))
        X = X_input
        X = Reshape(input_shape)(X)

        X = LocallyConnected1D(32, (8),  strides=(4), activation="relu",
                   padding="valid", kernel_initializer='he_uniform')(X)  
                   #try time-distrubuted    
        X = LocallyConnected1D(64, (4), strides=(3), activation="relu",
                   padding="valid", kernel_initializer='he_uniform')(X)  

        X = LocallyConnected1D(64, (2), strides=(2), activation="relu",
                   padding="valid", kernel_initializer='he_uniform')(X)             
        X = Dropout(.3)(X)
        X = Flatten()(X)
        X = Dense(self.network_size,  activation="relu",
                  kernel_initializer='he_uniform')(X)
        X = Dense(self.network_size,  activation="relu",
                  kernel_initializer='he_uniform')(X)

        
        if dueling:
            state_value = Dense(1,kernel_initializer='he_uniform',activation="softmax")(X)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(action_space,kernel_initializer='he_uniform', activation="linear")(X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",kernel_initializer='he_uniform')(X)

        model = Model(inputs = X_input, outputs = X, name = 'LocallyConnected1D')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate),  metrics=["accuracy"])
        
        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
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
       
    
        if self.memory.__len__() < self.batch_size:
            return
        minibatch = random.sample(self.memory.data[0:self.memory.__len__()], self.batch_size)
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
        # i = states.flatten() 

                # Create FileWriter
         

        # history = self.model.fit(states, targets_full, verbose=0)
        self.model.fit(states, targets_full, verbose=0)
        # global gl_loss
        # gl_loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):

            self.scores.append(score)
            self.episodes.append(episode)
            self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
            pylab.plot(self.episodes, self.average, 'r')
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            global log_data
            log_data.append((episode,score))
            try:
                pylab.savefig( self.modelname+"_CNN.png")
            except OSError:
                pass

            return str(self.average[-1])[:5]


        
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# g=  tf.get_default_graph()
# file_writer = tf.summary.FileWriter(log_dir, g)    
gl_total_frames=0
gl_score =0
gl_loss=0

def train_dqn(episode,  graphics=True, ch=300,  lchk = 0 , model=None):    
    loss = []     
    action_space = 6
    state_space = 4*138
    max_steps = 98*9
    # agent =ic(DQN(action_space, state_space,  model=ic(keras.models.load_model('CNN-990--0.40.h5'))))
    # for e in range(990,episode):
    agent =DQN(action_space, state_space,  model=model)
    agent.env_name = "StarShip"
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
            if i !=0:
                if i % 98==0: 
                   env.time_multipliyer *= 1.4                 
                if i % 4 ==0: 
                    env.reward +=  env.time_multipliyer * 0.04 + i *1e-7
                    env.time_multipliyer += 1e-4

          

            if (env.save):
                agent.saveModel(score)
                env.save = False
            action = agent.act(state)
            reward, next_state, done = env.stepNew(action)
            next_state = env.getEnvStateOnScreen()#do i need this? 
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
            
            # Add values to Tensorboard
          
        
            average  = 0
            if done:
                
                average = agent.PlotModel(score,e)
                print("episode: {}/{}, score:  {:0.3f}, average: {}".format(e, episode, score, average))
                # print("Max: ",i," Ep: ",e)
                # # print("episode: {}/{}, score: {}, lr : {}".format(e,global file_writer   t
                # training_summary = tf.Summary(value=[
                #     tf.Summary.Value(tag="loss", simple_value=gl_loss),
                #     tf.Summary.Value(tag="average", simple_value=float(average)),
                #     tf.Summary.Value(tag="score", simple_value=score),
                #     tf.Summary.Value(tag="max-step", simple_value=i),
                #     tf.Summary.Value(tag="dead obstacles", simple_value=env.obstacleGenerator.deadObstacles)
                #     ])          
                
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
            agent.saveModel(score)
    agent.saveModel(score)


    def test(self):
            for e in range(self.EPISODES):
                state = env.resetNew()
                done = False
                i = 0
                while not done:
                    action = np.argmax(self.model.predict(state))
                    next_state, reward, done, _ = env.stepNew(action)
                    i += 1
                    if done:
                        print("episode: {}/{}, score: {}".format(e, self.EPISODES))
                        break

log_data =[]
def plot_loss(ep, loss): 
    try: 
        plt.plot([i for i in range(ep)], loss)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.title('Episodal Loss')
        plt.savefig('logs\\loss_plot.png')          
        DQN.saveLog(log_data,'logs\\loss_plot.txt')

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
    train_dqn(ep)
