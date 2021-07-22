# ship
Route Optimisation in Ships
1. Q-Learning (without considering ships direction)
2. Deep Q-learning (without considering ships direction)
3. Deep Q-learning (with considering ships direction)

The environment is based on a grid world. 

Parameters considered within the envirnoment:
- Boundary conditions 
- Weather conditions (modelled to be stationary/nonchanging)
- Euclidean distance of current state wrt to the goal compared to preivous state wrt the goal (to be able to model how close the agent is)
- direction of ship/maneuvering 
