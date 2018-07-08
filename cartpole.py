"""
File: cartpole.py
Created: 2017-03-06
By Peter Caven, peter@sparseinference.com
Description:

-- Python 3.6 --

Solve the CartPole-v0 problem:
- observed features are expanded with a random transform to ensure linear separability.
- action selection is by dot product of an expanded observation with a weight vector.
- a queued history of recent observations is shuffled and replayed to update the output weights.
- output weights are updated at the end of each incomplete episode by Widrow-Hoff LMS update.
- the target outputs for the LMS algorithm are the means of the past outputs.
- output weights are maintained at a fixed norm for regularization.


"""

import gym
from gym import wrappers

from numpy import *
from numpy.random import uniform,normal
from numpy.linalg import norm

from random import shuffle
from collections import deque
from statistics import mean

env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, '../experiments/cartpole-experiment-1')


#------------------------------------------------------------------
# Hyperparameters

alpha = 1.0e-1              # the 'learning rate'
maxEpisodes = 1000          # run the agent for 'maxEpisodes'
maxTimeSteps = 500          # maximum number of steps per episode
fixedNorm = 0.5             # output weights are scaled to have norm == 'fixedNorm'
maxHistory = 2500           # maximum number of recent observations for replay
solvedEpisodes = 100        # cartpole is solved when average reward > 195 for 'solvedEpisodes'

#------------------------------------------------------------------
# Observations Transform

inputLength = 4             # length of an observation vector
expansionFactor = 30        # expand observation dimensions by 'expansionFactor'
expandedLength = expansionFactor*inputLength  # length of transformed observations

# Feature transform with fixed random weights.
V = normal(scale=1.0, size=(expandedLength, inputLength))

# Output weights, randomly initialized.
W = uniform(low=-1.0, high=1.0, size=expandedLength)
# Fix the norm of the output weights to 'fixedNorm'.
W *= fixedNorm/norm(W)

#------------------------------------------------------------------

def CartPoleAgent(alpha, W, V):
    """
    CartPoleAgent solves 'CartPole-v0'.
    """
    #--------------------------------------------------
    # observation history
    H = deque([], maxHistory)
    # episode total reward history
    R = deque([], solvedEpisodes)
    # histories of positive and negative outputs
    PO = deque([0], maxHistory)
    NO = deque([0], maxHistory)
    #--------------------------------------------------
    for episode in range(maxEpisodes):
        observation = env.reset()
        H.append(observation)
        totalReward = 0
        for t in range(1,maxTimeSteps+1):
            env.render()
            #--------------------------------------------------
            out = dot(tanh(dot(V,observation)), W)
            if out < 0:
                NO.append(out)
                action = 0
            else:
                PO.append(out)
                action = 1
            #--------------------------------------------------
            observation, reward, done, info = env.step(action)
            H.append(observation)
            totalReward += reward
            #--------------------------------------------------
            if done:
                R.append(totalReward)
                if t < 200:
                    #------------------------------------------
                    # Replay shuffled past observations using the 
                    # latest weights.
                    # Use the means of past outputs as 
                    # LMS algorithm target outputs.
                    #------------------------------------------
                    mn = mean(NO)           
                    mp = mean(PO)
                    shuffle(H)
                    for obs in H:
                        h = tanh(dot(V,obs))       # transform the observation
                        out = dot(h, W)
                        if out < 0:
                            e = mn - out
                        else:
                            e = mp - out
                        W += alpha * e * h          # Widrow-Hoff LMS update
                        W *= fixedNorm/norm(W)      # keep the weights at fixed norm
                    #------------------------------------------
                #--------------------------------------------------
                avgReward = sum(R)/solvedEpisodes
                print(f"[{episode:3d}:{totalReward:3.0f}] R:{avgReward:6.2f} mp:{mean(PO):7.3f} mn:{mean(NO):7.3f}  len(H):{len(H):4d}  W:{W[:2]}", flush=True)
                #--------------------------------------------------
                if avgReward == 200:
                    print("Solved.")
                    return
                #--------------------------------------------------
                break
        #------------------------------------------------------------------
    #------------------------------------------------------------------



CartPoleAgent(alpha, W, V)
env.close()


