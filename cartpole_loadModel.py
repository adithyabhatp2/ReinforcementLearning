import random
import gym
import numpy as np
import sys
import os
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import RMSprop

def printUsage():
    print "Usage: python cartpole.py <pathToModel> <numEpisodes>"

if len(sys.argv) < 3:
    print("Not enough arguments: " + str(len(sys.argv)))
    printUsage()
    sys.exit()

np.random.seed(7)
pathToModel = sys.argv[1]
EPISODES = int(sys.argv[2])


if __name__ == "__main__":
    # The .env at the last removes the time limit of 200 timesteps
    env = gym.make('CartPole-v0').env
    state_size = env.observation_space.shape[0] #* env.observation_space.shape[1] * env.observation_space.shape[2]
    
    game_model = load_model(pathToModel)

    sum = 0

    for e in xrange(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(1000):
            # env.render()
            act_values = game_model.predict(state)
            action = np.argmax(act_values[0])

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            state = next_state
            
            if done or time == 999:
                print("episode:\t{}\ttime:\t{}\t"
                        .format(e, time))
                sum+=time
                break

    print("Total\t{}\tAvg\t{}".format(sum, sum/EPISODES))
