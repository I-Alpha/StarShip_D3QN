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
from tensorflow.keras.layers import Bidirectional, LSTM
from keras.layers import Dense, LayerNormalization, LeakyReLU, DepthwiseConv2D,Embedding ,  Lambda,  Add, Average, TimeDistributed, Conv1D, Conv2D, Subtract, Activation, LocallyConnected2D, LocallyConnected1D, Reshape, concatenate, Concatenate, Flatten, Input, Dropout, MaxPooling1D,  MaxPooling2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from StarShip import StarShipGame
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
from keras.layers.advanced_activations import PReLU,LeakyReLU


HUBER_LOSS_DELTA = 1 


def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)
 
def gaussian(x):
    return K.exp(-K.pow(x,2))

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def build_1CNNBase(self, action_space=6, dueling=True):
        self.network_size = 256
        X_input = Input(shape=(self.REM_STEP*54,))
        input_reshape=(self.REM_STEP,54)
        X = X_input
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()

        const_init = initializers.constant(1e-2)
    
        X = X_input 
        X = Reshape(input_reshape)(X)    

        cnn1 = TimeDistributed(Dense(64, activation =PReLU(), kernel_initializer='he_uniform',))(X) 
        cnn1 = TimeDistributed(Dense(64, activation =PReLU(),kernel_initializer='he_uniform',))(cnn1) 
        cnn1 = LocallyConnected1D(filters=64,  kernel_size=4, activation =PReLU(),kernel_initializer='he_uniform')(cnn1)
        cnn1 = Flatten()(cnn1)
        cnn1 = Dense(512,activation="relu", kernel_initializer='he_uniform', )(cnn1)
        cnn1 = Dense(256,activation="relu", kernel_initializer='he_uniform', )(cnn1) 

        # cnn2 = TimeDistributed(Dense(64, kernel_initializer='he_uniform',))(X) 
        # cnn2 = LeakyReLU(.4)(cnn2)
        # cnn2 = MaxPooling1D(2)(cnn2) 
        # cnn2 = Flatten()(cnn2) 

        # cnn2 = LocallyConnected1D(filters=64, kernel_initializer='he_uniform', kernel_size=2)(X)
        # cnn2 = LeakyReLU(0.3)(cnn2)
        # cnn2 = LocallyConnected1D(filters=64, kernel_initializer='he_uniform', kernel_size=2)(X)
        # cnn2 = Dense(128,activation="relu", kernel_initializer='he_uniform', )(cnn2)
        # cnn2 = Flatten()(cnn2)

        cnn3 = LocallyConnected1D(filters=32, kernel_initializer='he_uniform', kernel_size=3)(X)
        cnn3 = gelu(cnn3)
        cnn3 = Conv1D(filters=64, kernel_initializer='he_uniform', padding="same", kernel_size=2)(cnn3)        
        cnn3 = gelu(cnn3)
        cnn3 = MaxPooling1D(pool_size=(2))(cnn3)
        cnn3 = Dropout(0.15)(cnn3)
        cnn3 = Flatten()(cnn3)
        cnn3 = Dense(128,activation="relu", kernel_initializer='he_uniform', )(cnn3)
     
        merge = concatenate([cnn1,cnn3])

        X = Dense(self.network_size*2, 
                  kernel_initializer='he_uniform',activation ="relu")(merge) 
 
        X = Dense(self.network_size, 
                  kernel_initializer='he_uniform',activation ="relu")(X)
        if dueling:
            state_value = Dense(
                1, kernel_initializer='he_uniform' )(X)
            state_value = Lambda(lambda s: K.expand_dims(
                s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(
                action_space, kernel_initializer='he_uniform') (X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(
                a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",
                      kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='build_TMaxpoolin_3')
        model.compile(loss=huber_loss, optimizer=Adam(
            lr=self.learning_rate),  metrics=["accuracy"])

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model

def build_modelPar1(self, dueling=True):
        self.network_size = 256
        X_input = Input(shape=(self.REM_STEP*54,))
        input_reshape=(1,4,54)
        x = X_input
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()

        const_init = initializers.constant(1e-2)
        
        v = Reshape((4,-1))(x)
        t = Reshape(input_reshape)(x)
        cnn2 = (LocallyConnected2D(filters=64,  kernel_initializer='he_uniform', kernel_size=(1,4), strides =(1,1), padding = "valid"))(t)
        cnn2 = gaussian(cnn2)
        cnn2 = Flatten()(cnn2)
        cnn2 = Dense(128, activation="relu",kernel_initializer=x_init)(cnn2)         
 

        cnn1 = TimeDistributed(Dense(64,kernel_initializer='he_uniform',))(v)       
        cnn1 = LeakyReLU(.3)(cnn1)
        cnn1 = TimeDistributed(Dense(64, activation="relu", kernel_initializer='he_uniform',))(cnn1)       
        cnn1 = Flatten()(cnn1)
        cnn1 = (Dense(128, activation="relu", kernel_initializer='he_uniform',))(cnn1)       
        
        
        conc = concatenate([cnn1,cnn2])
        w = Dense(512, activation="relu",kernel_initializer=x_init)(conc)
        w = Dense(256, activation="relu",kernel_initializer=x_init)(w)        

        state_value = Dense(1, kernel_initializer='he_uniform')(w)
        state_value = Lambda(lambda s: K.expand_dims(
        s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        action_advantage = Dense(
        self.action_space, activation='linear', kernel_initializer='he_uniform')(w)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
        a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage])
  
        model = Model([x], out, name='TCnn-model_1')

        if self.optimizer_model == 'Adam':
            optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(self.learning_rate, 0.99, 0.0, 1e-6,)
        else:
            print('Invalid optimizer!')

        model.compile(loss="mse",  optimizer=optimizer)
        model.summary()
        return model

def build_Parrallel_64(self):
        self.network_size = 256
        X_input = Input(shape=(self.state_space,))
        input_reshape=((self.REM_STEP,-1))
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        
        x_init = "he_uniform"

        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)

    
        X = X_input  
        t = Reshape(input_reshape)(X_input)
 
        x = Dense(64,
                  kernel_initializer=x_init)(x)     
        x = gaussian(x)
        out_a = (x)

        x = TimeDistributed(Dense(64,
                  kernel_initializer=x_init))(t)     
        x = gaussian(x)
        x = MaxPooling1D(2)(x) 
        x = Flatten()(x) 
        out_b = (x) 
     
        
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

        concatenated = concatenate([out_a, out_b])
        # model_final.add(Reshape((4,11,2), input_shape=(88,)))
        # model_final.add(concatted)
        # model_final.add(Flatten())
        # model_final.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        # # model_final.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        # out_d=  Dropout(0.4)(concatenated)
        # out_d=  MaxPooling2D((8,1))(out_d)

        out_d = (concatenated)

        out_d = Dense(512, activation='relu',
                      kernel_initializer='he_uniform')(out_d)
        out_d = Dense(64, activation='relu',
                      kernel_initializer='he_uniform')(out_d)

        state_value = Dense(1, kernel_initializer='he_uniform')(out_d)
        state_value = Lambda(lambda s: K.expand_dims(
            s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        action_advantage = Dense(
            self.action_space, activation='linear', kernel_initializer='he_uniform')(out_d)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
            a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage])

        model_final = Model([X_input], out, name='Pa12_model')

        model_final.compile(loss=huber_loss, optimizer=RMSprop(self.learning_rate, 0.99, 0.0, 1e-6), metrics=["accuracy"])
        #  RMSprop(lr=self.learning_rate, rho=0.95, decay=25e-5, moepsilon=self.epsilon), metrics=["accuracy"])
        print(model_final.summary())
        return model_final

def mean(input):
        return K.mean(input, axis=1)

def logloss(y_true, y_pred):  # policy loss
        return -K.sum(K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1)
        # BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

    # loss function for critic output
def sumofsquares(y_true, y_pred):  # critic loss
        return K.sum(K.square(y_pred - y_true), axis=-1)


def FCTime_distributed_model(self, action_space=6, dueling=True):
        self.network_size = 256
        X_input = Input(shape=(self.REM_STEP*99) )
        # X_input = Input(shape=(self.REM_STEP*7,))
        input_reshape=(self.REM_STEP,99)
        X = X_input
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)        
        X_reshaped = Reshape(input_reshape)(X_input)


        #normailsation fort each dimension 
        normlayer_1 = LayerNormalization( 
                axis=2,trainable=True,epsilon =0.001,center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones") 
        normlayer_2 = LayerNormalization( 
                axis=1,trainable=True,epsilon =0.001,center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones")  

        slice2 = tf.slice(X_reshaped,[0,0,0],[-1,2,10]) 
        slice2 = Reshape((2,-1))(slice2)

        TD1 = (normlayer_1)(slice2)
        # TD1 = (normlayer_2)(TD1)
        TD1 = TimeDistributed(Dense(16, activation = "relu", kernel_initializer=y_init, use_bias=True))(TD1) 
        # TD1 = Reshape((2,-1))(TD1)        
        TD1 = Flatten()(TD1) 
        TD1 = Dense(64, activation = "relu", kernel_initializer=y_init, use_bias=True)(TD1) 
        TD1 = Dense(64, activation = "relu", kernel_initializer=y_init,use_bias=True )(TD1) 
        


        normlayer_1 = LayerNormalization( 
                axis=2,trainable=True,epsilon =0.1,center=False,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones") 
        normlayer_2 = LayerNormalization( 
                axis=1,trainable=True,epsilon =0.0001,center=False,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones")
        slice3 = tf.slice(X_reshaped,[0,0,10],[-1,2,4]) 
        slice3 = Reshape((2,-1))(slice3)

        TD2 = (normlayer_1)(slice3)
        # TD2 = (normlayer_2)(TD2)
        TD2 = TimeDistributed(Dense(16, activation ="relu", kernel_initializer=y_init,use_bias=True))(TD2) 
        # TD2 = Reshape((2,-1))(TD2)         
        TD2 = Flatten()(TD2)     
        TD2 = Dense(64, activation ="relu" ,kernel_initializer=y_init,use_bias=True )(TD2) 
        TD2 = Dense(64, activation ="relu" ,kernel_initializer=y_init ,use_bias=True)(TD2) 
        # 

        normlayer_1 = LayerNormalization( 
                axis=2,trainable=True,epsilon =0.1,center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones") 
        normlayer_2 = LayerNormalization( 
                axis=1,trainable=True,epsilon =0.0001,center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones")
        slice4 = tf.slice(X_reshaped,[0,0,14],[-1,2,25])
        slice4 = Reshape((2,-1))(slice4)

        TD3 = (normlayer_1)(slice4)
        # TD3 = (normlayer_2)(TD3)
        TD3 = TimeDistributed(Dense(16, activation ="relu", kernel_initializer=y_init,use_bias=True ))(TD3) 
        # TD3 = Reshape((2,-1))(TD3)
        TD3 = Flatten()(TD3) 
        TD3 = Dense(64, activation ="relu",kernel_initializer=y_init,use_bias=True )(TD3) 
        TD3 = Dense(64, activation ="relu",kernel_initializer=y_init, use_bias=True)(TD3) 
        # 
  
        slice5 = tf.slice(X_reshaped,[0,0,39],[-1,2,-1])
        slice5 = Reshape((2,-1))(slice5)

        normlayer_1 = LayerNormalization( 
                axis=2,trainable=True,epsilon =0.1,center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones") 
        normlayer_2 = LayerNormalization( 
                axis=1,trainable=True,epsilon =0.0001,center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones")

        TD4 = (normlayer_1)(slice5)
        TD4 = (normlayer_2)(TD4)
        TD4 = TimeDistributed(Dense(8, activation = "relu" ,kernel_initializer=y_init,use_bias=True))(TD4) 
        # TD4 = Reshape((2,-1))(TD4)        "relu", 
        TD4 = Flatten()(TD4) 
        TD4 = Dense(64, activation ="relu", kernel_initializer=y_init,use_bias=True )(TD4)
        TD4 = Dense(64, activation ="relu", kernel_initializer=y_init,use_bias=True )(TD4)  
        # 
        # R1=LSTM(64)(X_reshaped)# cnn1 = Conv1D(filters=64,  kernel_size=(2),activation =LeakyReLU(.4),kernel_initializer=y_init,use_bias=True)(cnn1)          
        
        concatenated = concatenate([TD4,TD3,TD2,TD1])
        cnn1 = Flatten()(concatenated) 
        cnn1 = Dense(256,kernel_initializer=y_init, activation ="relu",use_bias=True )(cnn1)     
        cnn1 = Dense(256,kernel_initializer=y_init, activation =PReLU(),use_bias=True )(cnn1)
        cnn1 = Dense(128,kernel_initializer=y_init, activation ="relu",use_bias=True )(cnn1)     
        X= cnn1
 
                         
        # X = Conv2D(64, 4, strides=(2),padding="valid",activation="elu", kernel_initializer=x_init,   data_format="channels_first")(X)
        # X = Conv2D(128, 4, strides=(2),padding="valid",activation="elu",kernel_initializer=x_init,   data_format="channels_first")(X) 3cnn
        # 'Dense' is the basic form of a neural network layer
        # # Input Layer of state size(4) and Hidden Layer with 512 nodes
        # X = Dense(256, activation="relu", kernel_initializer=x_init)(X)
        # X = Dense(64, activation="relu", kernel_initializer=x_init)(X)
        # # X = Dense(64, activation="relu", kernel_initializer=x_init)(X)
        # X = Dense(256, activation="relu", kernel_initializer=const_init,use_bias=True, bias_initializer=truncatedn_init)(X) 
        # # Hidden layer with 256 nodes 

        # # Hidden layer with 64 nodes
        # X = Dense(64, activation="relu", kernel_initializer=truncatedn_init, bias_initializer=const_init)(X)
        if dueling:
            state_value = Dense(1, kernel_initializer=y_init, activation="softmax")(X)
            state_value = Lambda(lambda s: K.expand_dims(
                s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(
                action_space, kernel_initializer=y_init, activation="linear" )(X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(
                a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",
                      kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='Hydra-1')
        model.compile(loss=huber_loss, optimizer=Adam(
            lr=self.learning_rate),  metrics=["accuracy"])

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model

def build_Embedded(self, action_space=6, dueling=True):
        self.network_size = 256      
        X_input = Input(shape=(self.REM_STEP*99) )
        # X_input = Input(shape=(self.REM_STEP*7,))
        input_reshape=(self.REM_STEP,99)
        X = X_input
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)        
        X_reshaped = Reshape(input_reshape)(X_input)

 

        X = LayerNormalization(axis=2)(X_reshaped)
        X = LayerNormalization(axis=1)(X)
        X = Reshape((2,-1))(X)       
        X = Dense(self.network_size*2, kernel_initializer=y_init, activation ="relu")(X)
        X = Dense(256, kernel_initializer=y_init, activation ="relu")(X)
        X = TimeDistributed(Dense(self.network_size/4, kernel_initializer=y_init,activation ="relu"))(X)
        X = Flatten()(X)  
        if dueling:
            state_value = Dense(
                1, activation ="softmax")(X)
            state_value = Lambda(lambda s: K.expand_dims(
                s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(
                action_space, activation ="linear") (X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(
                a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",
                      kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='build_Embedded')
        model.compile(loss=huber_loss, optimizer=Adam(
            lr=self.learning_rate),  metrics=["accuracy"])

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model

def build_LSTM(self, action_space=6, dueling=True):
        self.network_size = 256
        X_input = Input(shape=(self.REM_STEP*112,))
        input_reshape=(self.REM_STEP,112,)
        X = X_input
        Xo = Reshape(input_reshape)(X) 
        
        # X = Dense(self.network_size/4, 
        #           kernel_initializer='he_uniform',activation ="relu")(X)
        # X1 = Flatten()(X) 

         
        # X = Dense(self.network_size/4, 
        #           kernel_initializer='he_uniform',activation ="relu")(X)
        # X2 = Flatten()(X) 
        
        # concatenated = concatenate([X1,X2])
        # # model_final.add(Reshape((4,11,2), input_shape=(88

        X = Dense(512, 
                  kernel_initializer='he_uniform')(Xo)
        X = LeakyReLU(0.7)(X) 
        X = LSTM(128,recurrent_activation="tanh", activation="relu", kernel_initializer="he_uniform")(X)
        if dueling:
            state_value = Dense( 
                1, kernel_initializer='he_uniform' ,activation="softmax")(X)
            state_value = Lambda(lambda s: K.expand_dims(
                s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(action_space,activation="linear", kernel_initializer='he_uniform') (X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(
                a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",
                      kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='Base_FC-LSTM_128n')
        model.compile(loss=huber_loss, optimizer=Adam(
            lr=self.learning_rate),  metrics=["accuracy"])

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model