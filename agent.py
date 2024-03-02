import numpy as np

# TODO: update descriptions for class and all functions

class Agent():
    """TODO: description"""
    
    def __init__(self, pos, delta):
        """Initializes the Agent object with an initial position
        
        Arguments:
            pos (nparray): 2x1 numpy array containing initial position of the agent
            delta (float): communication/sensing range of agent
        """
        self.pos = pos
        self.delta = delta
        self.neighbors = {}
        
    def update(self):
        """4th-order Runge-Kutta method to propogate agent dynamics. Calls the `derivatives` function to calculate derivatives of the state.
        
        Arguments:
            neighbors (list): List of Agent objects that influence the dynamics of the current agent
        
        """
        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = self.derivatives(self.pos)
        k2 = self.derivatives(self.pos + self.Ts/2.*k1)
        k3 = self.derivatives(self.pos + self.Ts/2.*k2)
        k4 = self.derivatives(self.pos + self.Ts*k3)
        self.pos += self.Ts/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    def derivatives(self):
        """Calculates the derivative of the state (equivalent to the control law, in this case) according to the neighbors of the agent
        
        """
        u = 0
        for key in self.neighbors:
            u -= self.dVij(self.neighbors[key])
            
        return u
    
    def dVij(self, agent_j):
        """Derivative of the edge-tension function. This defines the influence agent_j has on the rendezvous control law to ensure
        that the communication graph of all the agents remains connected."""
        l_ij_norm = np.linalg.norm(self.pos - agent_j.pos)
        return (2*self.delta - l_ij_norm)/(self.delta - l_ij_norm)**2*(self.pos - agent_j.pos)