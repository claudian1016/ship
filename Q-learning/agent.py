
###to fix OMP issue###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#import torch

import numpy as np
#import gym
import random
import time

import pickle  # to save/load Q-Tables
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


from new_env_final import ShipEnv

env = ShipEnv() 

#print(env.observation_space) #will print out: Discrete(16)
S_n = env.observation_space.n 
#print(S_n) #will yield 16

#print(env.action_space) #will print out: Discrete(nS)
A_n = env.action_space.n  
#print(A_n) #will yield number of actions

#creating the Q-Table + initializing all the Q-values to zero for each state-action pair

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

#print(q_table)

#initializing Q_Learning Parameters + TRAINING the agent

SHOW_EVERY = 10

num_episodes = 1000
#we define the total number of episodes we want the agent to play during training
max_steps_per_episode = 25
#define a maximum # of steps that our agent is allowed to take within a single episode

learning_rate = 0.6
#alpha
discount_rate = 0.9
#gamma

#exploration/exploitation trade-off in regards to the epsilong-greedy policy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01 #change it from 0.01 to 0.001 

#create this list to hold all of the rewards we'll get from each episode (this list is empty as of the beginning)
#this will be so we can see how our game score changes over time
rewards_all_episodes = [] 
rewards_all_cummulative = []

# Q-learning algorithm
for episode in range(num_episodes):
    
    # initialize new episode params  
    state = env.reset() 

    #done variable just keeps track of whether or not our episode is finished
    #so we initialize it to False when we first state the episode --> later we will see where it will get updated to notify us when the episode is over
    done = False
    
    #we enter into the nested loop, which runs for each time-step within an episode
    #exploration vs exploitation
    
    rewards_current_episode = 0
    rewards_cummulative = 0
        
    #we need to keep track of the rewards within the current episode as well, so we set it to 0 since we start out with no rewards at the beginning of each episode
            
    while done == False:

        for step in range(max_steps_per_episode): 
            #exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            
            if exploration_rate_threshold > exploration_rate:
                #if the threshold is greater than the exploration rate (which is initially set to 1), agent will exploit the env --> chose action that will result in highest Q-value
                action = np.argmax(q_table[state,:])
                #"argmax" returns you the index of the maximum value in the array
            else:
                action = env.action_space.sample() #exploring the env and chosing action at random
                #print(action)
                #for each time-step within an episode, we set our exploration rate_thrshold to a random number between 0 and 1
                #^will be used to determine whether our agent will explore or exploit the environment in this time-step#
                        
                #taking action
                new_state, reward, done, info = env.step(action) 
                print(reward)
                print(new_state)
                print(done) 
                
                #after our action is chosen, we take that action by calling step() on our env object and passing our action to it
                #the function step() returns a tuple containing the new state, the reward for the action, and info regarding our environment 
                        
                #update Q-table for Q(s,a) --> Q-value function in code form
                q_table[state,action] = q_table[state,action]*(1-learning_rate)+learning_rate*(reward + discount_rate * np.max(q_table[new_state, :]))
                        
                #transition to the next state
                state = new_state
                rewards_current_episode = reward
                rewards_cummulative += reward
                
                #print(rewards_current_episode)
                # we set our current state to the new_state that was returned to us once we took our last action
                #update the rewards from our current episode by adding the reward we recieved for our previous action
                        
                if done == True:
                    #did the action end the episode? if it did end, then we jump back to the beginning and move onto the next episode. Otherwise we moveonto the next timestep within the same episode
                    break
    
            #exploratoin rate decay
            #once an episode is finished, we need to update our exploration_rate using exponential decay

            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
                        
            rewards_all_episodes.append(rewards_current_episode)   
            rewards_all_cummulative.append(rewards_cummulative)  
                 
            #we then just append the rewards from the current episode to the list of rewards from all episodes 
            #print(rewards_all_episodes)
            
        #after all episodes complete  --> calculate the average reward per thousand episodes from our list that contains the rewards for all episodes so that we cna print it out                   
        #Calculate and print the average reward per thousand episodes
        rewards_per_thousand_episodes = np.array_split(np.array(rewards_all_episodes),num_episodes/1)
                
        #count = 1000
        #print("********Average reward per thousand episodes********\n")
        #for r in rewards_per_thousand_episodes:
            #print(count, ": ", str(sum(r/1000)))
            #count += 1000
                    
        #Print updated Q-table
        #print("\n\n********Q-table********\n")
        #print(q_table)

"""
plt.figure(dpi=1200)
plt.tight_layout()
plt.plot([i for i in range(len(rewards_all_episodes))], rewards_all_episodes)
plt.ylabel("Reward (all episodes)")
plt.xlabel("step #")
plt.savefig("plots_dim5/ep" + str(num_episodes) + "step" + str(max_steps_per_episode))
plt.show()

plt.figure(dpi=1200)
plt.tight_layout()
plt.plot([i for i in range(len(rewards_all_cummulative))], rewards_all_cummulative)
plt.ylabel("Reward (cummulative)")
plt.xlabel("step #")
plt.savefig("plots_dim5/ep" + str(num_episodes) + "step" + str(max_steps_per_episode))
plt.show()

plt.figure(dpi=1200)
plt.tight_layout()
plt.plot([i for i in range(len(rewards_all_episodes))], np.cumsum(rewards_all_episodes))
plt.ylabel("Reward (cummulative)")
plt.xlabel("step #")
plt.savefig("plots_dim5/ep" + str(num_episodes) + "step" + str(max_steps_per_episode))
plt.show()

#np.savetxt("tables/dim50/rewards_ep" + str(num_episodes) + "step" + str(max_steps_per_episode) + ".csv", rewards_all_episodes, delimiter=',')
#np.savetxt("tables/dim50/q_table_ep" + str(num_episodes) + "step" + str(max_steps_per_episode) + ".csv", q_table, delimiter=',')
####np.savetxt("tables/dim10/average_reward_per1000_episodes_ep" + str(num_episodes) + "step" + str(max_steps_per_episode) + ".csv", rewards_per_thousand_episodes, delimiter=',')
"""
