import numpy as np
from graph import Graph
import matplotlib.pyplot as plt
from scipy.integrate import ode

t_start = 0
t_end = 2.
Ts = 0.01
delta = 4.
epsilon = 0.05
    
class Dynamics():
    def __init__(self, x0, delta, epsilon):
        self.graph = Graph(x0, delta, epsilon)
        self.delta = delta
        self.epsilon = epsilon
        self.state = x0
        self.n, self.dim = x0.shape
    
    def plot(self):
        plt.figure()
        plt.scatter(self.state[:, 0], self.state[:, 1])
        plt.axis('equal')
        plt.xlim((-10, 10))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.graph.neighbors_graph[i, j]:
                    plt.plot([self.state[i, 0], self.state[j, 0]], [self.state[i, 1], self.state[j, 1]], color = 'r')
                
        plt.show(block = False)
    
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
                if self.graph.neighbors_graph[i, j]:
                    u[i] -= self.dVij(state[i], state[j])

        return u.flatten()

def update(state, neighbors_graph):
    # Integrate ODE using Runge-Kutta RK4 algorithm
    k1 = derivatives(state, neighbors_graph)
    k2 = derivatives(state + Ts/2.*k1, neighbors_graph)
    k3 = derivatives(state + Ts/2.*k2, neighbors_graph)
    k4 = derivatives(state + Ts*k3, neighbors_graph)
    state += Ts/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return state

# x0 = np.array([[-8., 0],
#                [-6, 2],
#                [-6, -2],
#                [-4, 0],
#                [0, 0],
#                [4, 0],
#                [6, 2],
#                [6, -2],
#                [8, 0]])

x0 = np.array([[-7.9, 0],
               [-5.925, 1.975],
               [-5.925, -1.975],
               [-3.95, 0],
               [0., 0],
               [3.95, 0],
               [5.925, 1.975],
               [5.925, -1.975],
               [7.9, 0]])


system = Dynamics(x0, delta, epsilon)


system.plot()
r = ode(system.derivatives).set_integrator("dopri5")
r.set_initial_value(system.state.flatten(), t=t_start)

while r.successful() and r.t < t_end:
    state = r.integrate(r.t + Ts)
    system.update_state_and_graph(state)
    
    if np.isclose(r.t, 0.08) or np.isclose(r.t, 0.4) or np.isclose(r.t, 1.6):
        system.plot()
    
system.plot()
input("Press Enter to exit")
    
    

    
    
    