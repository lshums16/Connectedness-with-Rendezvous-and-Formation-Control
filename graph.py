import numpy as np

# TODO: comment/document class and functions

class Graph():
    def __init__(self, agents, delta, epsilon):
        self.delta = delta
        self.epsilon = epsilon
        self.n = len(agents)
        self.dist_graph = np.zeros((self.n, self.n))
        self.delta_graph = np.zeros((self.n, self.n))
        self.neighbors_graph = np.zeros((self.n, self.n))
        
        self.update(agents)
        
    def update(self, agents):
        self.update_dist_graph(agents)
        self.update_delta_graph()
        self.update_neighbors_graph(agents)
        
    def update_dist_graph(self, agents):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.dist_graph[i, j] = np.linalg.norm(agents[i].pos - agents[j].pos)
                self.dist_graph[j, i] = self.dist_graph[i, j]
                
    def update_delta_graph(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.delta_graph[i, j] == 0:
                    if self.dist_graph[i, j] <= self.delta:
                        self.delta_graph[i, j] = 1
                        self.delta_graph[j, i] = 1
                else:
                    if self.dist_graph[i, j] > self.delta:
                        raise ValueError("Graph disconnected, fix control law")
                
                    
    def update_neighbors_graph(self, agents):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.neighbors_graph[i, j] == 0:
                    if self.dist_graph[i, j] <= (self.delta - self.epsilon):
                        agents[i].neighbors[j] = agents[j]
                        agents[j].neighbors[i] = agents[i]
                        self.neighbors_graph[i, j] = 1
                        self.neighbors_graph[j, i] = 1
                
                    
