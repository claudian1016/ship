###to fix OMP issue###
#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np

import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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
#print(states)
#print(S_n)

A_n = env.action_space.n  
actions = A_n

#action_space_size = env.action_space.n
#state_space_size = env.observation_space


def create_model(states, actions):
    learning_rate = 0.01
    #init = tensorflow.keras.initializers.HeUniform()
        
    model = Sequential()
    #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(8, input_shape = states, activation = 'relu'))
    #model.add(Dense(8, input_shape = states, activation = 'relu', kernel_initializer = init ))
    model.add(Dense(8, activation = 'relu'))
    #model.add(Dense(8, activation = 'relu', kernel_initializer = init ))
    model.add(Dense(actions, activation = 'linear')) 
    #model.add(Dense(actions, activation = 'linear', kernel_initializer = init ))
            
    #loss and optimizer function
        #loss = huber (can also Use MeanSquaredError())
        #optimizer = adam
    #model.compile(loss = tensorflow.keras.losses.Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        
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
dqn.test(env, nb_episodes = 15, visualize = True)
    

###reloading agent from memory 
dqn.save_weights('dqn_weights.h5f', overwrite= True)
states = env.observation_space.shape[0]
actions = env.action_space.n
model = create_model(states, actions)
dqn = DQNAgent(model = model, memory = memory, policy = policy, nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)
dqn.compile(Adam(learing_rate = 1e-3), metrics = ['mae'])
dqn.load_weights('dqn_weights.h5f')
dqn.test(env, nb_episodes = 5, visualize = True)
 
#_ = dqn.test(env, nb_episodes = 5, visualize = True)
   

mode = 'test'

"""
if mode == 'train':
    filename = 'trained_+_saved_agent_wo_direction'
    hist = dqn.fit(env, nb_steps=300000, visualize=False, verbose=2)
    with open('C:/Users/claudianori/Thesis_Codes/final/history_dqn_test_'+ filename + '.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # After training is done, we save the final weights.
    dqn.save_weights('Thesis_Codes/final/h5f_files/dqn_{}_weights.h5f'.format(filename), overwrite=True)
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)
"""
"""

if mode == 'test':
    #env.set_test_performace()  # Define the initialization as performance test
    #env.set_save_experice()  # Save the test to plot the results after
    filename = 'testing_wo_direction'
    dqn.load_weights('Thesis_Codes/final/h5f_filmost_recent/wo_direction_DQN_BoxObs/dqn_{}_weights.h5f'.format(filename))
    dqn.test(env, nb_episodes=10, visualize=True)
"""    
