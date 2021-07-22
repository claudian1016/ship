
import numpy as np
from numpy.random import RandomState

import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from stable_baselines3.common.env_checker import check_env 

DETERMINISTIC = False #outcome/result of running the code will change each time that it is run

class ShipEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # gym specific render function
    # human --> output will be something like SFFF,FFHF, etc with a color tag of current observation

    # Define constants for cleaner code
    size = int(input("What are the dimensions? "))

    # actions possible by agent
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    LEFT_DOWN = 4
    LEFT_UP = 5
    RIGHT_DOWN = 6
    RIGHT_UP = 7

    def __init__(self):
        super(ShipEnv, self).__init__()

        # grids --> currently will use grid_information as a reference
        self.grid_information = np.ones((self.size+2, self.size+2))
        # boundaries grid information
        self.grid_information[0, :] = "0"
        self.grid_information[:, 0] = "0"
        self.grid_information[-1, :] = "0"
        self.grid_information[:, -1] = "0"
        self.grid_information = self.grid_information*2
        # startpoint
        self.grid_information[1, 1] = 1
        # endpoint
        self.grid_information[-2, -2] = 3
        #print(self.grid_information)
        
        #rewards weather grid
        self.random_state_weather = RandomState(1234567890)
        self.rewards_weather_matrix = self.random_state_weather.randint(-2,1, size = (self.size+2, self.size+2))  
        
        # Shape of the 2D-grid
        self.determine = DETERMINISTIC
        desc = self.grid_information
        self.desc = desc = np.asarray(desc)  # np.asarray(desc, dtype='c')
        desc = desc.astype(int)
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.row = 1  # row
        self.col = 1  # col

        self.prevrow = 1
        self.prevcol = 1

        # Define starting and end point
        self.state = np.where(desc == 1)  # starting state = self.desc[1][1]
        # end state ---> or np.where(desc == 3) ?
        self.end_state = int(self.desc[-2][-2])

        nA = 8  # number of actions --> action space: up, down, left, right, up_left, up_right, down_left, down_right
        # number of states (IF considering "direction", change nS = nrow*ncol*8)
        nS = nrow * ncol

        # Define action_space
        self.action_space = spaces.Discrete(nA)  # 8
        self.actions = ['LEFT', 'DOWN', 'RIGHT', 'UP', 'LEFT_DOWN', 'LEFT_UP', 'RIGHT_DOWN', 'RIGHT_UP']

        # Define observation_space
        self.observation_space = spaces.Discrete(nS)

        # reward
        self.reward_range = (-100, 100)

    def step(self, action):

        #action = self.actions

        self.nrow, self.ncol = nrow, ncol = self.desc.shape
        
        #if step == 0:
        #self.row = 1
        #self.col = 1
        
        def to_state(row, col):
            # Returns a number which represents the state that the agent is in
            return row*self.ncol + col

        def movement(self, row, col, action):
            newcol = self.col
            newrow = self.row
            # defines the movement of the agent in terms of state number, by returning row and col position of the state it is in
            if action == self.LEFT:
                newcol = max(col - 1, 0)
            elif action == self.DOWN:
                newrow = min(row + 1, nrow - 1)
            elif action == self.RIGHT:
                newcol = min(col + 1, ncol - 1)
            elif action == self.UP:
                newrow = max(row - 1, 0)
            elif action == self.LEFT_DOWN:
                newcol = max(col - 1, 0)
                newrow = min(row + 1, nrow - 1)
            elif action == self.LEFT_UP:
                newcol = max(col - 1, 0)
                newrow = max(row - 1, 0)
            elif action == self.RIGHT_DOWN:
                newcol = min(col + 1, ncol - 1)
                newrow = min(row + 1, nrow - 1)
            elif action == self.RIGHT_UP:
                newcol = min(col + 1, ncol - 1)
                newrow = max(row - 1, 0)
                
            print("Action: " + self.actions[action])
            print("New row: " + str(newrow) + ", new col: " + str(newcol))
            return (newrow, newcol)  # returns row and col of agent position

        # obtain newrow, newcol by passing in row,col,action into the movement method
        newrow, newcol = movement(self, self.row, self.col, action)
        # obtain new state by passing in newrow and newcol into to_state() method
        newstate = to_state(newrow, newcol)
        # pass newrow and newcol into shape of the grid which will return grid position (defined by value) for new state
        new_value = self.desc[newrow, newcol]

        ###METHOD: consideration of ships distance (from current position to end goal)###
        
        def get_distance_method(row, col, newrow, newcol):

            reward_grid_distance = np.ones((self.size+2, self.size+2))*1000

            x_goal = self.size - 1
            y_goal = self.size - 1

            for i in range(self.size):
                for j in range(self.size):
                    reward_grid_distance[i+1][j+1] = np.sqrt(np.square(y_goal-j)+np.square(x_goal-i))

            return reward_grid_distance

        def get_reward_value(self):

            global reward

            self.b = get_distance_method(self.prevrow, self.prevcol, self.row, self.col)

            self.prevcol = self.col
            self.prevrow = self.row

            self.row = newrow
            self.col = newcol

            # rewards in consideration of BOUNDARY + START/END position of ship

            if int(new_value) == 0:
                reward = -10
            elif int(new_value) == 1:
                reward = -5
            elif int(new_value) == 2:
                reward = 0
            elif int(new_value) == 3:  # in goal position
                reward = 100

            #print("Terrain cost (either 0, 1, 2, or 3): " + str(new_value))
            #print("Reward after position: " + str(reward))

            # [[1000.         1000.         1000.         1000.         1000.     1000.         1000.        ]
            # [1000.          5.656          5.           4.472         4.123     4.            1000.        ]
            # [1000.            5.          4.24          3.605         3.162     3.            1000.        ]
            # [1000.            4.47        3.60          2.82          2.236     2.            1000.        ]
            # [1000.            4.12        3.16          2.23          1.41      1.            1000.        ]
            # [1000.            4.            3.            2.            1.      0.            1000.        ]
            # [1000.          1000.         1000.         1000.         1000.     1000.         1000.        ]]

            # rewards in consideration of distance of ships current position wrt the goal
            if self.b[self.prevrow][self.prevcol] > self.b[self.row][self.col]:
                reward += 1  # if ship is moving towards the goal
            elif self.b[self.prevrow][self.prevcol] < self.b[self.row][self.col]:
                reward += -1  # if ship is moving away from goal

            print("Previous distance: " + str(self.b[self.prevrow][self.prevcol]))
            print("Next distance: " + str(self.b[self.row][self.col]))
            print("Reward after distance: " + str(reward))

            # rewards in consideration of weather grid
            if self.rewards_weather_matrix[self.row][self.col] == -2:
                reward += -5
            elif self.rewards_weather_matrix[self.row][self.col] == -1:
                reward += -1
            elif self.rewards_weather_matrix[self.row][self.col] == 0:
                reward += 0

            print("Weather: " + str(self.rewards_weather_matrix[self.row][self.col]) + " (from matrix)")
            print("Reward after weather: " + str(reward))

            return reward

        reward = get_reward_value(self)

        done = int(new_value) == 3 or int(new_value) == 0

        while done == True:
            break

        # print(action)
        #print(newcol, newrow)
        # print(int(new_value))
        # print(newstate)
        #print("Reward: " + str(reward))
        print("-----")
        # print(done)

        info = {}

        return int(newstate), reward, done, info

        super(ShipEnv, self).step(movement(self, self.row, self.col, action))
             
    def reset(self): 
        self.grid_information = self.grid_information
        desc = self.grid_information
        self.desc = desc = np.asarray(desc, dtype='c')
        #desc = desc.astype(int) 
        self.nrow, self.ncol = nrow, ncol = desc.shape 
        
        self.row = 1  # row
        self.col = 1  # col

        self.prevrow = 1
        self.prevcol = 1
        
        #Initialize the agent at the starting point
        self.state = self.desc[1,1]  #starting state
        self.end_state = self.desc[-1][-1]  #end state

        return int(self.state)

        #here we convert to float32 to make it more general (in case we want to use continuous actions)
            #return np.array([self.agent_pos]).astype(np.float32)

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        
    def close(self):
        pass

#check whether gym environment is valid
env = ShipEnv() 
check_env(env, warn=True) 

#observation_space = env.observation_space
#print("Observation space:", env.observation_space) #gives Discrete(49)...= 7x7 (being the dimension of the grid/map)
#print("Shape:", env.observation_space.shape) 
    #the reason why shape is not "printed" is because we are dealing with a discrete space. 
    #in mathematical terms, a discrete space is one in which the points form a discontinuous sequence, meaning they are isolated from each other - doesnt have a 'shape'
#action_space = env.action_space #gives Discrete(8) for the 8 actions
#print(env.action_space) 

#action = action_space.sample()
#print(action)
#data = env.step(action) 
#print(len(data)) #length of data = 4 (considers the following 4 outputs of the step function: newstate, reward, done, info)

# The reset method is called at the beginning of an episode
#obs = env.reset()
# Sample a random action
#action = env.action_space.sample()
#print("Sampled action:", action)

#obs, reward, done, info = env.step(action, step) 
    # Note the obs is a numpy array --> the observation in the case of MY environment points to the STATE value that an individual grid space holds (this SHOULD change each time the code is run)
    # info is an empty dict ... can contain any debugging info
    # reward is a scalar

#print(obs, reward, done, info)
    #obs --> the state value wrt most recent action gets printed
    #reward --> check
