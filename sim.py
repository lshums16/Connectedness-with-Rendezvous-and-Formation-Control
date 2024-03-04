import numpy as np
from graph import Graph
import matplotlib.pyplot as plt

Ts = 0.000005
delta = 4.
epsilon = 0.05

def plot(state, neighbors_graph):
    plt.figure()
    plt.scatter(state[:, 0], state[:, 1])
    plt.axis('equal')
    plt.xlim((-10, 10))
    for i in range(n):
        for j in range(i + 1, n):
            if neighbors_graph[i, j]:
                plt.plot([state[i, 0], state[j, 0]], [state[i, 1], state[j, 1]], color = 'r')
                
    plt.show(block = False)
    

def dVij(agent_i_pos, agent_j_pos):
        """Derivative of the edge-tension function. This defines the influence agent_j has on the rendezvous control law to ensure
        that the communication graph of all the agents remains connected."""
        l_ij_norm = np.linalg.norm(agent_i_pos - agent_j_pos)
        return (2*delta - l_ij_norm)/(delta - l_ij_norm)**2*(agent_i_pos - agent_j_pos)

def derivatives(state, neighbors_graph):
    n = len(state)
    u = np.zeros(np.shape(state))
    for i in range(n):
        for j in range(n):
            if neighbors_graph[i, j]:
                u[i] -= dVij(state[i], state[j])

    return u

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

n = len(x0)

graph = Graph(x0, delta, epsilon)



state = x0

plot(state, graph.neighbors_graph)

t = 0.
while True:
    state = update(state, graph.neighbors_graph)
    graph.update(state)
    t += Ts
    
    if np.isclose(t, 0.08) or np.isclose(t, 0.4) or np.isclose(t, 1.6):
        plot(state, graph.neighbors_graph)
        
    if t > 0.081:
        break
    
input("Press Enter to exit")
    
    

    
    
    