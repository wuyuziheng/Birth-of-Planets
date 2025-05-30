import torch 
import math 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float64
G = torch.tensor(6.674e-11, dtype=dtype)

class Init_Utils():
    """
        Arguments:
            num        --> int 
            pos_norm   --> float 
            pos_bias   --> float
            v_norm     --> float
            v_bias     --> float
    """
    def __init__(self, num_planets, pos_norm, pos_bias, v_norm, v_bias):
        self.num = num_planets
        self.pos_norm = pos_norm
        self.pos_bias = pos_bias
        self.v_norm = v_norm
        self.v_bias = v_bias

    def _get_max_v(self, sun_mass):
        # get the max speed if the orbit is a ellipse
        return math.sqrt(2*G*sun_mass/self.pos_norm)

    def init_M_state(self):
        # init the planets at the nearest point
        init_pos = torch.tensor([self.pos_norm, 0, 0]).unsqueeze(0) # initial position
        init_v = torch.tensor([0, self.v_norm, 0]).unsqueeze(0) # initial velocity
        M = torch.cat([init_pos+torch.normal(0, self.pos_bias, size=(self.num, 3)).to(dtype=dtype), init_v+torch.normal(0, self.v_bias, size=(self.num, 3)).to(dtype=dtype)], dim=1) # add a bias to the initial state
        return M

    def init_radii_state(self, basic_radii):
        # set the radius of planet 
        return torch.full((self.num,),basic_radii).to(dtype=dtype).abs() # We first assume they have the same radius at beginning

    def _ellipse_orbit(self, theta, v, phi, r, sun_mass):
        h = v*math.sin(phi)*r
        E = (1/2)*v*v - (G * sun_mass)/r
        e = math.sqrt(1+(2*E*h*h)/(G*G*sun_mass*sun_mass))
        return (h**2/(G*sun_mass))/(1+e*np.cos(theta))
    
    def _draw_polar(self, series: pd.Series, output_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')  
        ax.set_theta_direction(-1) 
        ax.set_theta_zero_location('N')  
        ax.plot(series.index, series.values, '--')
        plt.savefig(output_path)
        plt.close(fig)

    def draw_orbit(self, sun_mass, output_path):
        config = {"v": self.v_norm, "phi": math.pi/2, "r": self.pos_norm, "sun_mass": sun_mass}
        theta = np.arange(0, 2*math.pi, 0.01)
        orbit = pd.Series(self._ellipse_orbit(theta, **config), theta)
        self._draw_polar(orbit, output_path)
        