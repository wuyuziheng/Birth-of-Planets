import math
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from init_utils import *
from auto_config import *
import cv2

from moviepy import *
import os
from natsort import natsorted

"""
    Public Variables:
        device --> where to solve the ode (seems that cpu better than gpu, since the dimension of the matrix is not too large)
        dtype  --> type of data
        G      --> gravitational constant
        FPS    --> number of frames in a second 
        SIZE   --> resolution of the video
"""
# device = torch.device('cuda')
device = torch.device('cpu')
dtype = torch.float64
FPS = 30
SIZE = (640,480)

class Params():
    """
        A customized parameter parser.
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

class Func():
    """
        Compute the derivative of the current state.
    """
    def __init__(self, planet_mass, sun_mass):
        # The sun is always assumed to be centered at the origin.
        self.sun_mass = sun_mass
        self.planet_mass = planet_mass

    def __call__(self, M):
        # M: Each row (x, y, z, vx, vy, vz) is the state of a planet.
        assert M.size(1) == 6
        assert M.size(0) == self.planet_mass.size(0)
        num = M.size(0)
        M_with_sun = torch.cat([M, torch.tensor([[0,0,0,0,0,0]], dtype=dtype, device=device)], dim=0)
        mass_with_sun = torch.cat([self.planet_mass, torch.tensor([self.sun_mass], dtype=dtype, device=device)], dim=0)
        dist_vec_matrix = M_with_sun[:, :3].unsqueeze(0).repeat(num+1,1,1) - M_with_sun[:,:3].unsqueeze(1).repeat(1,num+1,1)
        dist_matrix = dist_vec_matrix.square().sum(dim=2).fill_diagonal_(1).sqrt()
        force_matrix = (G * dist_vec_matrix * (mass_with_sun.unsqueeze(1) @ mass_with_sun.unsqueeze(0)).unsqueeze(2).repeat(1,1,3) / dist_matrix.pow(3).unsqueeze(2).repeat(1,1,3))
        force_matrix = torch.stack([force_matrix[:,:,0].fill_diagonal_(0), force_matrix[:,:,1].fill_diagonal_(0), force_matrix[:,:,2].fill_diagonal_(0)], dim=2)
        force = force_matrix.sum(dim = 1)[:-1,:]
        a = force / self.planet_mass.unsqueeze(1).repeat(1,3)
        return torch.cat([M[:,3:6], a], dim=1)
        

def RK5_one_step(func, time, M):
    """
        Runge-Kutta 5 ODE solver.
    """
    # Moving the model forward by time using 1-stepped Runge-Kutta5.
    k_1 = time * func(M)
    k_2 = time * func(M + k_1 * (1/4))
    k_3 = time * func(M + k_1 * (3/32) + k_2 * (9/32))
    k_4 = time * func(M + k_1 * (1932/2197) - k_2 * (7200/2197) + k_3 * (7296/2197))
    k_5 = time * func(M + k_1 * (439/216) - k_2 * 8 + k_3 * (3680/513) - k_4 * (845/4104))
    k_6 = time * func(M - k_1 * (8/27) + k_2 * 2 - k_3 * (3544/2565) + k_4 * (1859/4104) - k_5 * (11/40))
    return M + k_1 * (16/135) + k_3 * (6656/12825) + k_4 * (28561/56430) + k_5 * (-9/50) + k_6 * (2/55)

class Model():
    # The model
    def __init__(self, M, rho, radii, sun_mass, beta, time, step_num, total_num):
        """
        parameters:
            M: Initial state (position+velocity) of the planets. torch.tensor with shape num * 6, each row (x, y, z, vx, vy, vz) is the state of one planet.
            rho: Density of the planets.
            radii: Radii of the planets.
            sun_mass: Mass of the sun. The sun is always assumed to be located at the origin.
            beta: Merge threshold.
            time: The time of each step.
            step_num: Number of total steps.
        """
        assert M.size(0) == radii.size(0)
        self.M = M # the current state (pos, vel)
        self.rho = rho # density
        self.radii = radii # radius 
        self.sun_mass = sun_mass # mass of sun
        self.beta = beta # merging threshold
        self.time = time # time per step
        self.step_num = step_num # number of steps

        self.func = Func(torch.tensor(self.rho * (4/3) * math.pi, dtype=dtype, device=device) * self.radii.pow(3), self.sun_mass)
        self.num = M.size(0) # number of planets
        self.init_num = M.size(0)

        self.result_list = [] # cache of result
        self.radii_list = [] # cache of radii
        self.num_list = [] # cache of numbers of planets
        self.figure_list = [] # cache of figures

        self.video = None # a temp video of each chunks
        self.total_num = total_num # number of steps of the whole process (of all chunks)
        self.finished_chunks = 0 # chunks that already finished

        self.total_num_list = []

    def _reset(self):
        # reset caches, start the next chunk
        self.result_list = []
        self.radii_list = []
        self.total_num_list += self.num_list
        self.num_list = []
        self.figure_list = []
        self.video = None
        self.finished_chunks += 1
        
    def move_one_step_forward(self):
        # use current state to predict next state
        self.M = RK5_one_step(self.func, self.time, self.M)
        
    def merge(self):
        # check if merger occurred
        while(True):
            if self.num == 1:
                break
            temp1 = 0
            dist_matrix = (self.M[:,:3].unsqueeze(0).repeat(self.num,1,1) - self.M[:,:3].unsqueeze(1).repeat(1,self.num,1)).square().sum(-1).sqrt()
            threshold_matrix = self.beta * (self.radii.unsqueeze(-1) + self.radii.unsqueeze(-1).reshape(1,-1))
            tmp_matrix = torch.where(dist_matrix<=threshold_matrix, 1, 0).to(device)
            if tmp_matrix.sum() == self.num:
                break 
            else:
                tmp_num = self.M.size(0)
                for i in range(tmp_num): 
                    for j in range(i+1, tmp_num):
                        if tmp_matrix[i][j] == 1:
                            temp1 = 1
                            mi, mj = self.radii[i].pow(3), self.radii[j].pow(3)
                            pi, pj = self.M[i,:3], self.M[j,:3]
                            vi, vj = self.M[i,3:6], self.M[j,3:6]
                            new_position = (pi * mi + pj * mj) / (mi + mj)
                            new_velocity = (mi * vi + mj * vj) / (mi + mj)
                            new_radius = (mi + mj).pow(1/3)
                            self.M = torch.cat([self.M[:i,:], self.M[i+1:j,:], self.M[j+1:,:], torch.cat([new_position, new_velocity]).unsqueeze(dim=0)], dim=0)
                            self.radii = torch.cat([self.radii[:i], self.radii[i+1:j], self.radii[j+1:], new_radius.unsqueeze(dim=0)])
                            self.func = Func(torch.tensor(self.rho * (4/3) * math.pi, dtype=dtype, device=device) * self.radii.pow(3), self.sun_mass)
                            self.num = self.num - 1
                            break
                    if temp1 == 1: 
                        break

    def evolve(self):
        # prediction
        with tqdm(total=self.step_num) as pbar:
            self.merge()
            for i in range(self.step_num):
                self.move_one_step_forward()
                self.merge()
                self.result_list.append(self.M)
                self.radii_list.append(self.radii)
                self.num_list.append(self.num)
                # if self.num == 1:
                #     print("Merge Successed!")
                #     return True
                pbar.update(1)
        print(f"Merge Failed! Remaining {self.num} planets")
        return False

    def _draw_figure(self, i, M, radii, num, basic_radii, figure_range):
        # draw a single frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection ='3d')
        M_x = M[:,0].cpu().numpy()
        M_y = M[:,1].cpu().numpy()
        M_z = M[:,2].cpu().numpy()
        scale = 10*math.pi*radii.square()/(basic_radii**2)
        scale = scale.cpu().numpy()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(figure_range["length"][0], figure_range["length"][1])
        ax.set_ylim(figure_range["width"][0], figure_range["width"][1]) 
        ax.set_zlim(-figure_range["length"][1], figure_range["length"][1])
        ax.set_title(f"Process:{int(100*(self.finished_chunks*self.step_num+i+1)/self.total_num)}%, Remaining:{num}")
        ax.scatter(0, 0, 0, c='y', marker='o')
        ax.scatter(M_x, M_y, M_z, s=scale, c='r', marker='o') 
        plt.savefig("./image.png")
        plt.close(fig)
        self.video.write(cv2.imread("./image.png"))
        os.remove("./image.png")
        
    def generate_video(self, record_steps, basic_radii, output_path, figure_range):
        # merge to a video
        with tqdm(total=int(self.step_num/record_steps)) as pbar:
            self.video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE) 
            for i in range(0, self.step_num, record_steps):
                self._draw_figure(i, self.result_list[i], self.radii_list[i], self.num_list[i], basic_radii, figure_range)
                pbar.update(1)
            self.video.release()
            cv2.destroyAllWindows()

    def draw_distribution_graph(self, output_path):
        fig = plt.figure()
        x = np.arange(self.total_num)
        y = np.array(self.total_num_list)
        plt.xlim((0,self.total_num))
        plt.ylim((0,self.init_num))
        plt.plot(x, y, 'r--')
        plt.savefig(output_path)
        plt.close(fig)

if __name__ == "__main__":
    """
        Arguments:
            M                  --> init position of planets
            rho                --> density of the planets
            radii              --> radii of planets
            sun_mass           --> mass of the system center 
            merging_threshold  --> merging threshold
            per_step_time      --> time per step
            num_steps          --> number of steps
            num_planets        --> number of planets
            init_config        --> config for initialization
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/config.json")
    args = parser.parse_args()

    # Parse arguments ---> Add auto config to 
    config = Params(args.config_path).dict
    auto = AutoConfig(config)
    auto_config = config["auto_config"]
    method = auto_config["method"]
    AUTO_CONFIG_MAP = {
        "id-length": auto.identity_length,
        "id-mass": auto.identity_mass,
        "id-time": auto.identity_time,
        "id-scope": auto.identity_scope,
        "metric-mode": auto.init_config_from_metrics
    }
    try:
        output_path = AUTO_CONFIG_MAP[method](**auto_config["config"])
        print(f"----------Change to Auto Config {output_path}----------")
        config = Params(output_path).dict
    except:
        print("----------Invalid auto config setting! Use the original config ... ----------")
    G = config["G"]
    rho = config["rho"]
    sun_mass = config["sun_mass"]
    beta = config["merging_threshold"]
    step_num = config["num_steps"]
    time = config["per_step_time"]
    basic_radii = config["basic_radii"]
    num_chunks = config["num_chunks"]
    G = torch.tensor(G, dtype=dtype).to(device)
    init_state = Init_Utils(G, **config["init_config"])
    total_num = num_chunks * step_num
    M = init_state.init_M_state().to(device)
    radii = init_state.init_radii_state(basic_radii).to(device)
    

    # compute the range of velocity
    tmp_max = init_state._get_max_v(sun_mass)
    tmp_min = (1/math.sqrt(2))*tmp_max
    print("Max initial velocity:", tmp_max)
    print("Min initial velocity:", tmp_min)
    if init_state.v_norm < tmp_min or init_state.v_norm > tmp_max: 
        print("----------WARNING!!! This velocity may be invalid!----------")

    # make sure the traget directory exist
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")
    if not os.path.exists("./result"):
        os.mkdir("./result")

    # initialize the figure config
    fig_config = config["figure_config"]
    print("Orbit figure saved.")
    init_state.draw_orbit(sun_mass, "./result/orbit.png")
    margin_bias = fig_config["margin_bias"]
    max_figure_range = torch.quantile(init_state.get_figure_range(M, sun_mass), fig_config["range_quantile"]) + margin_bias
    min_figure_range = init_state.pos_norm + margin_bias
    width = (max_figure_range + min_figure_range)/2
    figure_range = {"length":[-max_figure_range, min_figure_range], "width":[-width, width]}
    record_steps = fig_config["record_steps"]

    # Run Model in chunks
    model = Model(M, rho, radii, sun_mass, beta, time, step_num, total_num)
    for i in range(num_chunks):
        print(f"----------Predicting ... Chunk ({i+1}/{num_chunks})----------")
        if model.evolve():
            break
        print(f"----------Generating video ... Chunk ({i+1}/{num_chunks})----------")
        model.generate_video(record_steps, basic_radii, f"./tmp/image-{i}.mp4", figure_range)
        model._reset()
    print("Successfully predicted!")
    model.draw_distribution_graph("./result/distibution.png")

    # merge videos in each chunk
    L = [] 
    
    for root, dirs, files in os.walk("./tmp/"):
        files=natsorted(files) 
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                filePath = os.path.join(root, file)
                video = VideoFileClip(filePath)
                os.remove(filePath)
                L.append(video)

    if len(L) != 0:
        final_clip = concatenate_videoclips(L)
        final_clip.write_videofile("./result/final.mp4", fps=FPS, remove_temp=False)
    else:
        print("----------WARNING!!! No Videos Found!----------")
