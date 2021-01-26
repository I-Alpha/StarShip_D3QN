import argparse
from DQN_agent import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from icecream import ic
# shutdown gpu
# import os
# import tensorflow as tf

# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# # if tf.test.gpu_device_name():
# #     print('GPU found')
# # else:
# #     print("No GPU found")


parser = argparse.ArgumentParser(
    description='Args for initialising training and testing of RL Agent')
parser.add_argument('--mode', '-m', default='train')
parser.add_argument('--model', '-mp', default=None)
parser.add_argument('--epochs', '-e', type=int,default=2000)
parser.add_argument('--learnrate', '-l', default=.001)
parser.add_argument('--plot', '-p', default=True)
parser.add_argument('--graphics', '-g', default=True)
parser.add_argument('--checkpoint', '-chk',type=int, default=None)
parser.add_argument('--last_epoch', '-le', type=int, default=-1) 

args = parser.parse_args()

 
if __name__ == '__main__': 
    if args.checkpoint == None:  
        args.checkpoint = 100
        print("\n\n\n --checkpoint not given.\n Default checkpoint set at {} epochs.".format(args.checkpoint))
    else :
        print("checkpoint at {}".format(args.checkpoint))
    if args.last_epoch > 0 and  args.model == None:
        print("\n\n\nError. No model specified but last checkpoint specified at ",args.last_epoch,".")
        exit()
    if args.mode == 'train':
            if args.model != None:
                try: 
                        model = keras.models.load_model(args.model)
                        print("\n  Training beginning....\n\n  Model  {  " + args.model + "  }  has been loaded! \n\n Calling train function.....\n\n\n\n")                       
                except :
                        print("Error.Check model is not incompatible with current enviroment configuration")
                        exit()
                try:
                    loss = train_dqn( args.epochs,args.graphics, ch=args.checkpoint, lchk=args.last_epoch+1, model=model)
                except:
                    print("Something went wrong with training. Check model is not incompatible with current enviroment configuration")
                    exit()
            else:
                    print(' \n No model specified. \n Building new model for training.\n\n')
                    loss = train_dqn(args.epochs, args.graphics, ch=args.checkpoint)

    if args.plot:
                plt.plot([i for i in range(args.epochs)], loss)
                plt.xlabel('episodes')
                plt.ylabel('reward')
                plt.show()

    else:
        pass
