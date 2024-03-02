import numpy as np
from agent import Agent
from graph import Graph

delta = 4
epsilon = 0.05

x0 = np.array([[-8, 0],
               [-6, 2],
               [-6, -2],
               [-4, 0],
               [0, 0],
               [4, 0],
               [6, 2],
               [6, -2],
               [8, 0]])

n = len(x0)

agents = [None]*n
for i in range(n):
    agents[i] = Agent(np.array([x0[i]]).T, delta)

graph = Graph(agents, delta, epsilon)

while True:
    for i in range(n):
        agents[i].update()
    graph.update(agents)
    
    
    