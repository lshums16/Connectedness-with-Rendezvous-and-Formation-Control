import numpy as np

# TODO: comment/document class and functions

class Graph():
    def __init__(self, x0, delta, epsilon):
        self.delta = delta
        self.epsilon = epsilon
        self.n = len(x0)
        self.dist_graph = np.zeros((self.n, self.n))
        self.delta_graph = np.zeros((self.n, self.n))
        self.neighbors_graph = np.zeros((self.n, self.n))
        
        self.update(x0)
        
    def update(self, state):
        self.update_dist_graph(state)
        self.update_delta_graph()
        self.update_neighbors_graph()
        
    def update_dist_graph(self, state):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.dist_graph[i, j] = np.linalg.norm(state[i] - state[j])
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
                
                    
    def update_neighbors_graph(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.neighbors_graph[i, j] == 0:
                    if self.dist_graph[i, j] <= (self.delta - self.epsilon):
                        self.neighbors_graph[i, j] = 1
                        self.neighbors_graph[j, i] = 1
                
                    
