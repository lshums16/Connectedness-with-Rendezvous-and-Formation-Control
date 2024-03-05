import numpy as np
from scipy.integrate import ode
from multiagent_system import MultiAgentSystem

t_start = 0
t_end = 2.
Ts = 0.01
delta = 4.5
epsilon = 0.05

x0 = np.array([[-8., 0],
               [-6, 2],
               [-6, -2],
               [-4, 0],
               [0, 0],
               [4, 0],
               [6, 2],
               [6, -2],
               [8, 0]])

# x0 = np.array([[-7.9, 0],
#                [-5.925, 1.975],
#                [-5.925, -1.975],
#                [-3.95, 0],
#                [0., 0],
#                [3.95, 0],
#                [5.925, 1.975],
#                [5.925, -1.975],
#                [7.9, 0]])


system = MultiAgentSystem(x0, delta, epsilon, init_phase = "rendezvous_connected") # init_phase can be "rendezvous_connected" or "rendezvous_simple"

r = ode(system.derivatives).set_integrator("dopri5")
r.set_initial_value(system.state.flatten(), t=t_start)

while r.successful() and r.t < t_end:
    system.animate(r.t, Ts)
    state = r.integrate(r.t + Ts)
    system.update_state_and_graph(state)
    
    if np.max(system.graph.dist_graph) < 0.001:
        system.animate(r.t + Ts, Ts)
        break
    
input("Press Enter to exit")
    
    

    
    
    