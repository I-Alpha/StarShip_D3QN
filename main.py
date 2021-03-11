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
parser.add_argument('--model', '--mp', default=r"savedModels\Hydra-1\0309Mar\Hydra-1_epochs_4758_avg_-7.66_.h5" )
parser.add_argument('--epochs', '-e', type=int,default=10000)
parser.add_argument('--learnrate', '-l', default=.001)
parser.add_argument('--graphics', '-g', default=False)
parser.add_argument('--checkpoint', '-chk',type=int, default=None)
parser.add_argument('--last_epoch', '-le', type=int, default=-1) 

args = parser.parse_args()

 
if __name__ == '__main__': 
    if args.checkpoint == None:  
        args.checkpoint = 400
        print("\n\n\n --checkpoint not given.\n Default checkpoint set at {} epochs.".format(args.checkpoint))
    else :
        print("checkpoint at {}".format(args.checkpoint))
    if args.last_epoch > 0 and  args.model == None:
        print("\n\n\nError. No model specified but last checkpoint specified at ",args.last_epoch,".")
        exit()
        
    if args.mode == 'train':
            if args.model != None:
                try: 
                        with keras.utils.CustomObjectScope({'PReLU': PReLU,'LeakyReLU':LeakyReLU}):  
                            model = keras.models.load_model(args.model)
                        print(model.summary())       
                        print("\n\n\n  Model - {  " + args.model + "  }  has been loaded! ")                      
                        ans = input("Are you sure you want to use this model? Y/\/N ? :    ")
                        if ans.upper() != "Y":
                            b=print("Please rerun the program and choose a different model. Thanks. ")
                            exit(code=b)      
                        print("\nbeginning....Training \n\n ") 
                except :
                        print("Error. Unable to load model.")                   
                try:
                    loss = train_dqn( args.epochs,args.graphics, ch=args.checkpoint, lchk=args.last_epoch+1, model=model)
                except:
                    print("Something went wrong with training. Check model is not incompatible with current enviroment configuration")
                    exit()
            else:
                    print(' \n No model specified. \n Building new model for training.\n\n')
                    loss = train_dqn(args.epochs, args.graphics, ch=args.checkpoint)

    else:
        pass
