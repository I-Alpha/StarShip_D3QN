from keras.callbacks import History
from ring_buf import RingBuf
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

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)

def build_modelPar(self, dueling=True, input_shape=(4, 1, 138)):
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)
        if dueling:
            x = Input(shape=(self.state_space,))
            t = Reshape(input_shape)(x)
            # a series of fully connected layer for estimating V(s)
            y11 = Dense(128, activation='relu', kernel_initializer=truncatedn_init,
                        bias_initializer=const_init, use_bias=True)(t)
            y12 = Dense(128, activation='relu', kernel_initializer=truncatedn_init,
                        bias_initializer=const_init, use_bias=True)(y11)
            y13 = Flatten()(y12)
            y14 = Dense(self.action_space, activation="linear",
                        kernel_initializer=x_init)(y13)

            # a series of fully connected layer for estimating A(s,a)

            y20 = Flatten()(x)
            y21 = Dense(256, activation='relu', kernel_initializer=truncatedn_init,
                        bias_initializer=const_init, use_bias=True)(y20)
            y22 = Dense(128, activation='relu', kernel_initializer=truncatedn_init,
                        bias_initializer=const_init, use_bias=True)(y21)
            y23 = Dense(1, activation="linear", kernel_initializer=x_init)(y22)

            # a series of fully connected layer for estimating B(s,a)

            y30 = TimeDistributed(Dense(64, activation='relu'))(t)
            y31 = TimeDistributed(Dense(64, activation='relu'))(y30)
            y32 = Dense(1, activation='softmax')(y31)
            y33 = Flatten()(y32)
            y34 = Dense(256, activation='relu')(y33)
            y35 = Dense(64, activation='relu')(y34)
            y36 = Dense(self.action_space, activation='linear')(y35)

            w = Concatenate(axis=-1)([y23, y14])

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

def build_modelPar_2(self, input_shape=(4, 1, 138,)):

        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        truncatedn_init2 = initializers.TruncatedNormal(0, 2e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)
        const_init2 = initializers.constant(2e-2)
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
        out_a = (x)

        x = Dense(64, activation='relu',
                  kernel_initializer=x_init)(t)
        x = Dense(64, activation='softmax',
                  kernel_initializer='he_uniform')(x)
        out_b = (x)

        x = Dense(64, activation='relu',
                  kernel_initializer=y_init)(t)
        x = Dense(64, activation='softmax',
                  kernel_initializer='he_uniform')(x)
        out_c = (x)

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

        concatenated = concatenate([out_a, out_b, out_c])
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

        state_value = Dense(1, activation='softmax',
                            kernel_initializer='he_uniform')(out_d)
        state_value = Lambda(lambda s: K.expand_dims(
            s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        action_advantage = Dense(
            self.action_space, activation='linear', kernel_initializer='he_uniform')(out_d)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(
            a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)

        out = Add()([state_value, action_advantage])

        model_final = Model([digit_0], out, name='ParallelCNNmodel')

        model_final.compile(loss="mean_squared_error", optimizer=Adam(
            learning_rate=self.learning_rate), metrics=["accuracy"])
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

def build_model_lstm(self, input_shape=(336, 4), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(4*336,))
        X = X_input
        truncatedn_init = initializers.TruncatedNormal(0, 1e-2)
        x_init = "he_uniform"
        y_init = initializers.glorot_uniform()
        const_init = initializers.constant(1e-2)
        X = Reshape(input_shape)(X)
        # X =Conv1D(4,(4),(4),activation="relu",) (X)
        # X = Conv2D(1, (1,4), strides=(1,1),padding="same",activation="relu", kernel_initializer=x_init,   data_format="channels_first")(X)
        X = LSTM(4, return_sequences=True, return_state=True)(X)
        X = Flatten()(X)
        X = Dense(512, activation="relu", kernel_initializer=x_init)(X)
        X = Dense(256, activation="relu", kernel_initializer=x_init)(X)
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
            state_value = Dense(1, kernel_initializer=x_init)(X)
            state_value = Lambda(lambda s: K.expand_dims(
                s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(
                action_space, kernel_initializer=x_init)(X)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(
                a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

            X = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            X = Dense(action_space, activation="relu",
                      kernel_initializer='he_uniform', bias_initializer=const_init)(X)

        model = Model(inputs=X_input, outputs=X, name='3CNN_model')
        model.compile(loss="mean_squared_error", optimizer=Adam(
            lr=self.learning_rate),  metrics=["accuracy"])

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model

def build_Base(self, input_shape=(4,112), action_space=6, dueling=True):
        self.network_size = 256

        X_input = Input(shape=(4*112,))
        X = X_input
        X = Reshape(input_shape)(X)        
        Y = Reshape((4,112))(X_input)
        # X = Dense(64, kernel_initializer='he_uniform')(X)
        # X =LeakyReLU(alpha=.6)(X)
        # X = Dense(128, kernel_initializer='he_uniform')(X)
        # X =LeakyReLU(alpha=.1)(X)  
        # X = Conv1D(64, (2),  strides=(1), padding = "valid",
        #                kernel_initializer='he_uniform')(X)
        # X = LeakyReLU(alpha=.2)(X) 
        #  
        X = TimeDistributed(Dense(self.network_size/2))(X)   
        X= TimeDistributed(LeakyReLU(alpha=.3))(X)    
        X= Dense(self.network_size/4,activation="relu")(X)      
        out_a = Flatten()(X)
        
        X = Conv1D(128, (2) ,(1))(Y)         
        X = LeakyReLU(alpha=.3)(X)
        X = Conv1D(128, (2) ,(2))(X)    
        X = LeakyReLU(alpha=.3)(X)  
        X = Dense(self.network_size,activation="relu")(X)
        out_b = Flatten()(X)
        
        
        

        concatenated = concatenate([out_a, out_b])
        X = (concatenated) 


        X = Dense(self.network_size*2, 
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

        model = Model(inputs=X_input, outputs=X, name='Base1model')
        model.compile(loss=huber_loss, optimizer=Adam(
            lr=self.learning_rate),  metrics=["accuracy"])

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model