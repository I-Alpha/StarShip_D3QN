from keras.callbacks import History
import time
import datetime
from tensorflow.keras import initializers
import keras 
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
from itertools import chain, combinations
import csv
#Save funcrions 

def saveModel(obj, data, score="0",checkpoint =0):
    
        #save model as .h5 with png and loss history.txt        
        mkdir_p(obj.savedir)
        if obj.currEpisode < 1:
            return
        time_ = datetime.datetime.now
        time_h = time_().strftime("%h")      
        print("saving " + obj.model.name + "_epochs<" + str(obj.currEpisode) +">_"+str(int(score)) + "....")
        name = obj.model.name+ "_epochs_{}_avg_{:.2f}_".format(obj.currEpisode, data['average_score'].values[-1])
        try:
            obj.model.save(obj.savedir+name+".h5", overwrite=True)
            print(name + " saved! ")
        except:
            print(name + " not saved! ")
            return      
      
        # saveLog(obj, name+".txt", obj.savedir,autosavep=True)
       
   
def saveLog(obj, name="lastRun.txt",  dir="", autosavep =False):
    # Auto save for saving loss plot with txt
        mkdir_p(dir)
        if obj.currEpisode < 5:
            return
        try:
            f = open(dir+name, "w")
            for i, q in enumerate(obj.log_data):
                f.write("{},{}\n".format(i, q))
            f.close()
        except:
            print("Unable to write log.txt")
            return
        if autosavep:
            t1 =[ ]
            x1= []
            for i in obj.log_history:
                    i = i[0]
                    t1.append(i*-1)
                    x1.append(sum(t1)/len(obj.log_history))
            PlotData("Episode_versus_score",["Episode","score" ],[obj.log_data,obj.average],["score","average"] , obj.savedir)                      
            PlotData("Iteration_versus_loss",["Iteration","loss" ],[t1,x1],["loss","average"],obj.savedir)
            t2 =[]
            x2= []
            for i in obj.epsilon_log:
                    t2.append(i)
            PlotData("Episode_versus_Epsilon",["episode","epsilon" ],[t2],["Epsilon"],obj.savedir)       
         
   
#plot functions
figures = {}
plt.grid()


def PlotData(title,axeslabels=["Epsiodes","Loss"],values=[],labels=[],saveDir = ""):         
    #First value is x- value
    
        epochs = range(1, len(values[0])+1)
        for v,i in enumerate(values):
            plt.plot(epochs,i, label=labels[v])
        plt.grid()
        plt.xlabel(axeslabels[0])
        plt.ylabel(axeslabels[1])
        plt.legend()       
        plt.tight_layout()
        plt.savefig(saveDir+title+".png")
        plt.clf()


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
 
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

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