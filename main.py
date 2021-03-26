import argparse
from DQN_agent import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from icecream import ic
from Models import *
from keras.layers.advanced_activations import PReLU,LeakyReLU


parser = argparse.ArgumentParser(
    description='Args for initialising training and testing of RL Agent')
parser.add_argument('--mode', '-m', default='train')
parser.add_argument('--model', '--mp', default=None)# r"savedModels\Hydra-1\0309Mar\Hydra-1_epochs_4708_avg_-7.66_.h5" )
parser.add_argument('--epochs', '-e', type=int,default=10000)
parser.add_argument('--learnrate', '-l', default=.0001)
parser.add_argument('--graphics', '-g', default=True)
parser.add_argument('--checkpoint', '-chk',type=int, default=None)
parser.add_argument('--last_epoch', '-le', type=int, default=0) 

args = parser.parse_args()
model = args.model
if __name__ == '__main__': 
    if args.checkpoint == None:  
        args.checkpoint = 400
        print("\n-Checkpoint not given.\n-Default checkpoint set at {} epochs.".format(args.checkpoint))
    else :
        print("\n-Checkpoint at {}".format(args.checkpoint))
    if args.last_epoch > 0 and  args.model == None:        
        sys.exit("\nError. No model specified but last checkpoint specified at ",args.last_epoch,".")
    if args.model != None:
            try: 
                with keras.utils.CustomObjectScope({'PReLU': PReLU,'LeakyReLU':LeakyReLU}):  
                    model = keras.models.load_model(args.model)
                print(model.summary())       
                print("\n Model - {  " + args.model + "  }  has been loaded! ")                      
                ans = input("Are you sure you want to use this model? Y/\/N ? :    ")
                if ans.upper() != "Y":
                   sys.exit("Please load a different model. Thanks. ")
            except :
                sys.exit("Error. Unable to load model.")    


    if args.mode =="train"  :
        loss = train_dqn( args.epochs,args.graphics, ch=args.checkpoint, lchk=args.last_epoch, model=model)                        
    elif args.mode == 'test': 
        if args.model != None: 
            loss = test( args.epochs,args.graphics, model=model)
        else: 
            sys.exit("Please specify a model!")