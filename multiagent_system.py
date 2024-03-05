from graph import Graph
import matplotlib.pyplot as plt
import numpy as np

class MultiAgentSystem():
    def __init__(self, x0, delta, epsilon, init_phase = "rendezvous_connected"):
        self.graph = Graph(x0, delta, epsilon)
        self.delta = delta
        self.epsilon = epsilon
        self.state = x0
        self.n, self.dim = x0.shape
        self.phase = init_phase
        self.xlim = (np.min(x0[:, 0]), np.max(x0[:, 0]))
        self.ylim = (np.min(x0[:, 1]), np.max(x0[:, 1]))
    
    def animate(self, t, Ts):
        plt.clf()
        plt.scatter(self.state[:, 0], self.state[:, 1])
        plt.axis('equal')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.graph.neighbors_graph[i, j]:
                    plt.plot([self.state[i, 0], self.state[j, 0]], [self.state[i, 1], self.state[j, 1]], color = 'r')
        plt.figtext(0.8, 0.9, f"t = {t:.2f}s")
        plt.pause(Ts)
                
    
    def update_state_and_graph(self, state):
        self.state = state.reshape((self.n, self.dim))
        self.graph.update(self.state)
    
    def dVij(self, agent_i_pos, agent_j_pos):
        """Derivative of the edge-tension function. This defines the influence agent_j has on the rendezvous control law to ensure
        that the communication graph of all the agents remains connected."""
        l_ij_norm = np.linalg.norm(agent_i_pos - agent_j_pos)
        return (2*self.delta - l_ij_norm)/(self.delta - l_ij_norm)**2*(agent_i_pos - agent_j_pos)

    def derivatives(self, t, state):
        state = state.reshape(int(len(state)/self.dim), self.dim)
        u = np.zeros(np.shape(state))
        for i in range(self.n):
            for j in range(self.n):
                if self.phase == 'rendezvous_connected':
                    if self.graph.neighbors_graph[i, j]:
                        u[i] -= self.dVij(state[i], state[j])
                elif self.phase == 'rendezvous_simple':
                    if self.graph.delta_graph[i, j]:
                        u[i] -= (state[i] - state[j])

        return u.flatten()