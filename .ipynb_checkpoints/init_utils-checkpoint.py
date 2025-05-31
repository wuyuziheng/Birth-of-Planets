import torch 
import math 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float64
seed = 42
torch.manual_seed(seed)

class Init_Utils():
    """
        Arguments:
            num        --> int 
            pos_norm   --> float 
            pos_bias   --> float
            v_norm     --> float
            v_bias     --> float
    """
    def __init__(self, device, G, num_planets, pos_norm, pos_bias, v_norm, v_bias):
        self.G = G
        self.num = num_planets
        self.pos_norm = pos_norm
        self.pos_bias = pos_bias
        self.v_norm = v_norm
        self.v_bias = v_bias
        self.device = device

    def _get_max_v(self, sun_mass):
        # get the max speed if the orbit is a ellipse
        return math.sqrt(2*self.G*sun_mass/self.pos_norm)

    def init_M_state(self):
        # init the planets at the nearest point
        init_pos = torch.tensor([self.pos_norm, 0, 0], device=self.device).unsqueeze(0) # initial position
        init_v = torch.tensor([0, self.v_norm, 0], device=self.device).unsqueeze(0) # initial velocity
        M = torch.cat([init_pos+torch.normal(0, self.pos_bias, size=(self.num, 3), device=self.device).to(dtype=dtype), init_v+torch.normal(0, self.v_bias, size=(self.num, 3), device=self.device).to(dtype=dtype)], dim=1) # add a bias to the initial state
        return M

    def init_radii_state(self, basic_radii):
        # set the radius of planet 
        return torch.full((self.num,),basic_radii, device=self.device).to(dtype=dtype).abs() # We first assume they have the same radius at beginning

    def _ellipse_orbit(self, theta, v, sin_phi, r, sun_mass):
        h = v*sin_phi*r
        E = (1/2)*v*v - (self.G * sun_mass)/r
        e = torch.sqrt(1+(2*E*h*h)/(self.G*self.G*sun_mass*sun_mass))
        return (h**2/(self.G*sun_mass))/(1+e*torch.cos(theta))

    def _max_orbit_range(self, v, sin_phi, r, sun_mass):
        total_num = v.size(0)
        theta = torch.full((total_num,), math.pi, device=self.device)
        return self._ellipse_orbit(theta, v, sin_phi, r, sun_mass)
    
    def _draw_polar(self, series: pd.Series, output_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')  
        ax.set_theta_direction(-1) 
        ax.set_theta_zero_location('N')  
        ax.plot(series.index, series.values, '--')
        plt.savefig(output_path)
        plt.close(fig)

    def draw_orbit(self, sun_mass, output_path):
        config = {"v": self.v_norm, "sin_phi": math.sin(math.pi/2), "r": self.pos_norm, "sun_mass": sun_mass}
        theta = torch.arange(0, 2*math.pi, 0.01, device=self.device)
        orbit = pd.Series(self._ellipse_orbit(theta, **config).cpu().numpy(), theta.cpu().numpy())
        self._draw_polar(orbit, output_path)

    def get_figure_range(self, M, sun_mass):
        pos_state = M[:,:3]
        v_state = M[:,3:]
        r = pos_state.square().sum(-1).sqrt()
        v = v_state.square().sum(-1).sqrt()
        sin_phi = (pos_state * v_state).sum(-1)/(r*v + 1.0e-9) 
        sin_phi = (1 - sin_phi.square()).sqrt()
        return self._max_orbit_range(v, sin_phi, r, sun_mass)
        