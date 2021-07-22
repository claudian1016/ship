
###to fix OMP issue###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np

import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import pickle 
from new_env_final import ShipEnv

env = ShipEnv() 

S_n = env.observation_space 
states = S_n.shape

A_n = env.action_space.n  
actions = A_n

def create_model(states, actions):
    learning_rate = 0.01
    init = tensorflow.keras.initializers.HeUniform()
        
    model = Sequential()
    model.add(Dense(24, input_shape = states, activation = 'relu', kernel_initializer = init ))
    model.add(Dense(12, activation = 'relu', kernel_initializer = init ))
    model.add(Dense(actions, activation = 'linear', kernel_initializer = init ))
        
    #loss and optimizer function
        #loss = huber (can also Use MeanSquaredError())
        #optimizer = adam
    model.compile(loss = tensorflow.keras.losses.Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        
    return model
    
    del model

model = create_model(states, actions)    
print(model.summary())


#building agent
policy = BoltzmannQPolicy()
memory = SequentialMemory(limit = 50000, window_length = 1)
dqn = DQNAgent(model = model, memory = memory, policy = policy, nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)
dqn.compile(Adam(learning_rate=1e-3), metrics = ['mae'])
#dqn.fit --> trains the agent in the given environment
dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

    
scores = dqn.test(env, nb_episodes = 100, visualize = False)
print(np.mean(scores.history['episode_reward']))
#_ = dqn.test(env, nb_episode = 15, visualize = True)
