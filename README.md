# MDP-Planning

MDP-Planning problem is defined to be an agent environment interaction problem where in the agent has the knowledge of the next states ,transition probabilities and the reward functions for all the states present. We deploy algorithms to find the optimal Value and policy functions.
In the exps.py file there are three algorithms to solve the MDP Planning problem :
- Value Iteration
- Howard's Policy Iteration (with Policy Improvement)
- Linear Programming approach

The folder named mdp has sample mdp problems and solutions as a text file.
The folder named maze has sample grids and their solutions.

The files encoder.py and enc_w_ind.py convert a grid into an mdp with and without indexing respectively.
The mdp is then given as an input to either of the three algorithms. 
The solution case is input to decoder.py to represent policy as a path in the grid.
