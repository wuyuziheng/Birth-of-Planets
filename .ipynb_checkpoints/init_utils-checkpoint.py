import torch 
import math 

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
        