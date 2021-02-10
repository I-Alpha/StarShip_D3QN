# this is an example code on using the environment. 
import numpy as np
from StarShip import StarShipGame
import Utilities
# import the env class

# create an object of env class
env = StarShipGame(True)
np.random.seed(0)
log_data =[]
def random_policy(episode):

    action_space = 6
    state_space =66
    max_steps = 1000

    for e in range(episode):
        state = env.reset()
        score = 0

        for i in range(max_steps):
            action = np.random.randint(action_space)
            reward, next_state, done = env.step(action)
            score += reward
            state = next_state
            
            if done:
                log_data.append(score)
                Utilities.PlotData("epsiode vs loss",log_data)
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break


if __name__ == '__main__':

    random_policy(1000)
