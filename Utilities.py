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
from DQN_agent import *
#Save funcrions 

lastCheckpoint = 757


def saveModel(obj, score="n.a",checkpoint = 1):
        #save model as .h5 with png and loss history.txt


        if obj.currEpisode < 5:
            return
        time_ = datetime.datetime.now
        time_h = time_().strftime("%h")      
        print("saving " + obj.model.name + "-" +
              str(obj.currEpisode)+str(int(score)) + "....")
        name = obj.model.name+time_().strftime("%h") + \
            "_{}_{:0.2f}.h5".format(obj.currEpisode+lastCheckpoint-2, obj.average[-1])
        try:
            obj.model.save(obj.savedir+name, overwrite=True)
            print(name + " saved! ")
        except:
            print(name + " not saved! ")
            return
        try:
            saveLog(obj,name+".txt", obj.savedir,False)
            print(name + ".txt saved!")
        except:
            print("Error. Unable to save " +name  + ".txt")
            return
        try:
            pylab.savefig(obj.savedir + name+".png")
            print(name+".png saved!")
        except:
            print(name+".png not saved!")
            pass

def saveLog(obj, name="lastRun.txt", dir="logs/fit/", autosavep =True):
    # Auto save for saving loss plot with txt
        if obj.currEpisode < 5:
            return
        try:
            f = open(dir+name, "w")
            for i, q in obj.log_data:
                f.write("{},{}\n".format(i, q))
            f.close()
        except:
            print("Unable to write log txt")
            return
        if autosavep:
            try:
                pylab.savefig(dir+name+".png")
            except OSError:
                print("In-functio save failed for " + name + ".png. Continuing...")
                pass

#plot functions
pylab.figure(figsize=(18, 9))

def PlotModel(obj, score, episode):

        obj.scores.append(score)
        obj.episodes.append(episode)
        obj.average.append(sum(obj.scores[-50:]) / len(obj.scores[-50:]))
        pylab.plot(obj.episodes, obj.average, 'r')
        pylab.plot(obj.episodes, obj.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        obj.log_data.append((episode, score))
        try:
            pylab.savefig(obj.modelname+"_CNN.png")
        except OSError:
            pass

        return str(obj.average[-1])[:5],obj
 
def plot_loss(obj,ep, loss):
    try:
        plt.plot([i for i in range(ep)], loss)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.title('Episodal Loss')
        plt.savefig('logs\\loss_plot.png')
        saveLog(obj,obj.log_data, 'logs\\loss_plot.txt')

    except:
        print("loss_plot error. Skipping plotting.\n")
        pass


class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

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