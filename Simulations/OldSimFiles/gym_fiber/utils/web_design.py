import numpy as np
import gym
from elastica import *

class web_design():
    def __init__(self, simulator, web_origin, web_plane):
        self.web_origin = web_origin
        self.simulator = simulator

        if web_plane == 'xy':
            self.normal = np.array([0.0, 0.0, 1.0])
        elif web_plane == 'yz':
            self.normal = np.array([1.0, 0.0, 0.0])
        elif web_plane == 'zx':
            self.normal = np.array([0.0, 1.0, 0.0])
        else:
            print("Specify the plane of the Spider Web")

    def add_radial_threads(self, num_rad_threads):
        angle_between_threads = 2 * np.pi / num_rad_threads
        for i in range(num_rad_threads):
            ang = i * angle_between_threads
            direction = np.array([np.cos(ang), np.sin(ang), 0.0])
        pass
        
    def add_boundary_threads(self, num_bound_threads):
        pass

    def add_spiral_threads(self, num_spirals):
        pass