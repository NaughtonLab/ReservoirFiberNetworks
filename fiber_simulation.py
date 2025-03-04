import os
import pickle
import multiprocessing
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from collections import defaultdict

from elastica import *
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica.modules.damping import Damping

from utils.forces.pointforce import PointForce, PointForceSinsusoidal, PointForceSpline
from utils.networkcallback import NetworkCallBack
from utils.render.post_processing import plot_network_video, plot_network_video_2D

from tqdm import tqdm

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass

class fiber_simulation():

    def __init__(self, **kwargs):

        self.simulator = BaseSimulator()

        """NETWORK GEOMETRY"""
        self.num_horizontal_threads = kwargs.get("num_horizontal_threads", 2)
        self.num_vertical_threads = kwargs.get("num_vertical_threads", self.num_horizontal_threads)
        self.network_origin = kwargs.get("network_origin", np.zeros((3,)))

        """COMMON THREAD PROPERTIES"""
        self.thread_length = kwargs.get("thread_length", 1000e-3 * 1e3) # m --> 1e3 mm (1e3) 
        thread_diameter = kwargs.get("thread_diameter", 4e-3 * 1e3) # m --> 1e3 mm (1e3)
        self.thread_radius = 0.5 * thread_diameter
        self.dx = kwargs.get("dx", 20) # m --> 1e3 mm (1e3)
        self.n_elem = np.rint(self.thread_length/self.dx).astype(int)

        '''Young's modulus'''
        self.youngs_modulus = kwargs.get("youngs_modulus", 230e6) # Pa (N/m2) => kg / m / s2 --> 1e3 g / 1e3 mm / s2 (1.0) --> 1e6 mg / 1e3 mm / 1e6 ms2 (1e-3)
        # poisson_ratio = 0.45

        '''Density'''
        self.density = kwargs.get("density", 1000 * 1e-6) # kg/m3 --> 1e3 g / 1e9 mm3 (1e-6) --> 1e6 mg / 1e9 mm3 (1e-3)

        '''Damping Constant'''
        self.damping_constant = kwargs.get("damping_constant", 5)
        self.filter_order = kwargs.get("filter_order", 6)

        '''Tension Force holding the threads'''
        self.tension_force = kwargs.get("tension_force", 1e4)
        self.horizontal_tension_force = np.array([self.tension_force, 0.0, 0.0]) #1e3 --> 1 uN
        self.vertical_tension_force = np.array([0.0, self.tension_force, 0.0]) #1e3 --> 1 uN

        '''Magnitude and type of external point force'''
        self.point_force_mag = kwargs.get("point_force_mag", -2e5)
        self.SPREAD_PF = kwargs.get("SPREAD_PF", False)
        self.TYPE_PF = kwargs.get("TYPE_PF", "constant")

        """CONNECTION PARAMETERS"""
        self.k = kwargs.get("k", 1e9)
        self.nu = kwargs.get("nu", 0.0)
        self.kt = kwargs.get("kt", 1e9)

        """SIMULATION PARAMETERS"""
        '''Total Simulation time'''
        self.duration = kwargs.get("duration", 10)

        '''Physical simulation timestep'''
        self.sim_dt = kwargs.get("sim_dt", 5e-6) # ms

        '''Type of timestepper'''
        self.StatefulStepper = PositionVerlet()

        '''FPS for video'''
        self.rendering_fps = kwargs.get("rendering_fps", 1000)
        self.step_skip = np.rint(1.0 / (self.rendering_fps * self.sim_dt)).astype(int)

        self.normal = np.array([0.0, 0.0, 1.0])
        self.x_direction = np.array([1.0, 0.0, 0.0])
        self.y_direction = np.array([0.0, 1.0, 0.0])

        self.sample_freq = kwargs.get("sample_freq", 5)

        self.STOP_AT_NAN = kwargs.get("STOP_AT_NAN", True)
        self.CALLBACK = kwargs.get("CALLBACK", True)
        self.VIDEO = kwargs.get("VIDEO", True)


    def add_threads(self):
        """HORIZONTAL THREADS"""
        self.horizontal_thread = [None for i in range(self.num_horizontal_threads)]

        self.vert_connect_idx = np.linspace(0, self.n_elem+1, self.num_horizontal_threads+2)
        self.vert_connect_idx = self.vert_connect_idx[1:-1]
        for i in range(len(self.vert_connect_idx)):
            self.vert_connect_idx[i] = int(self.vert_connect_idx[i])
        horizontal_thread_y_pos = self.vert_connect_idx * self.dx - self.thread_length/2 + self.network_origin[1]

        for i in range(self.num_horizontal_threads):
            horizontal_thread_start = self.network_origin - 0.5 * self.thread_length * self.x_direction
            horizontal_thread_start[1] = horizontal_thread_y_pos[i]    

            self.horizontal_thread[i] = CosseratRod.straight_rod(
                n_elements=self.n_elem,
                start=horizontal_thread_start,
                direction=self.x_direction,
                normal=self.y_direction,
                base_length=self.thread_length,
                base_radius=self.thread_radius,
                density=self.density,
                youngs_modulus=self.youngs_modulus,
            )

            self.simulator.append(self.horizontal_thread[i])

            self.simulator.dampen(self.horizontal_thread[i]).using(
                AnalyticalLinearDamper, 
                damping_constant = self.damping_constant,
                time_step = self.sim_dt
            )

            self.simulator.dampen(self.horizontal_thread[i]).using(LaplaceDissipationFilter, filter_order = self.filter_order)

            self.simulator.constrain(self.horizontal_thread[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(0, ), 
                constrained_director_idx=(0, ), 
                translational_constraint_selector=np.array([False, True, True]),
                rotational_constraint_selector=np.array([True, True, False]),
            )

            self.simulator.constrain(self.horizontal_thread[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(-1, ), 
                constrained_director_idx=(-1, ), 
                translational_constraint_selector=np.array([False, True, True]),
                rotational_constraint_selector=np.array([True, True, False]),
            )

            self.simulator.add_forcing_to(self.horizontal_thread[i]).using(
                EndpointForces, -self.horizontal_tension_force, self.horizontal_tension_force, ramp_up_time=0.25
            )

        """VERTICAL THREADS"""
        self.vertical_thread = [None for i in range(self.num_vertical_threads)]
        self.hor_connect_idx = np.linspace(0, self.n_elem+1, self.num_vertical_threads+2)
        self.hor_connect_idx = self.hor_connect_idx[1:-1]
        for i in range(len(self.hor_connect_idx)):
            self.hor_connect_idx[i] = int(self.hor_connect_idx[i])
        vertical_thread_x_pos = self.hor_connect_idx * self.dx - self.thread_length/2 + self.network_origin[0]

        for i in range(self.num_vertical_threads):

            vertical_thread_start = np.array([vertical_thread_x_pos[i], self.network_origin[1]-self.thread_length/2, self.network_origin[2]])
            
            self.vertical_thread[i] = CosseratRod.straight_rod(
                n_elements=self.n_elem,
                start=vertical_thread_start,
                direction=self.y_direction,
                normal=self.x_direction,
                base_length=self.thread_length,
                base_radius=self.thread_radius,
                density=self.density,
                youngs_modulus=self.youngs_modulus,
            )

            self.simulator.append(self.vertical_thread[i])

            self.simulator.dampen(self.vertical_thread[i]).using(
                AnalyticalLinearDamper, 
                damping_constant = self.damping_constant,
                time_step = self.sim_dt
            )

            self.simulator.dampen(self.vertical_thread[i]).using(LaplaceDissipationFilter, filter_order = self.filter_order)

            self.simulator.constrain(self.vertical_thread[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(0, ), 
                constrained_director_idx=(0, ),
                translational_constraint_selector=np.array([True, False, True]), 
                rotational_constraint_selector=np.array([True, True, False])
            )

            self.simulator.constrain(self.vertical_thread[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(-1, ), 
                constrained_director_idx=(-1, ),
                translational_constraint_selector=np.array([True, False, True]), 
                rotational_constraint_selector=np.array([True, True, False])
            )

            self.simulator.add_forcing_to(self.vertical_thread[i]).using(
                EndpointForces, -self.vertical_tension_force, self.vertical_tension_force, ramp_up_time=0.25
            )

        """ADDING CONNECTIONS"""
        for j in range(self.num_horizontal_threads):
            for i in range(self.num_vertical_threads):
                rest_rotation_matrix = self.horizontal_thread[j].director_collection[..., int(self.hor_connect_idx[i])] @ self.vertical_thread[i].director_collection[..., int(self.vert_connect_idx[j])]
                self.simulator.connect(
                    self.horizontal_thread[j], self.vertical_thread[i],
                    int(self.hor_connect_idx[i]), int(self.vert_connect_idx[j])
                    ).using(FixedJoint,
                            k = self.k,
                            nu = self.nu,
                            kt = self.kt,
                            rest_rotation_matrix = rest_rotation_matrix
                            )

        """INITIALIZING CALLBACK"""
        if self.CALLBACK:            
            self.post_processing_dict_horizontal_thread = []
            for i in range(self.num_horizontal_threads):
                post_processing_dict_horizontal_thread_each = defaultdict(list)
                self.simulator.collect_diagnostics(self.horizontal_thread[i]).using(
                    NetworkCallBack,
                    step_skip=self.step_skip,
                    callback_params=post_processing_dict_horizontal_thread_each,
                )

                self.post_processing_dict_horizontal_thread.append(post_processing_dict_horizontal_thread_each)

            self.post_processing_dict_vertical_thread = []
            for i in range(self.num_vertical_threads):
                post_processing_dict_vertical_thread_each = defaultdict(list)
                self.simulator.collect_diagnostics(self.vertical_thread[i]).using(
                    NetworkCallBack,
                    step_skip=self.step_skip,
                    callback_params=post_processing_dict_vertical_thread_each,
                )

                self.post_processing_dict_vertical_thread.append(post_processing_dict_vertical_thread_each)

    def launch_sim(self):

        suffix = f'{self.duration:.0f}sec_L{self.thread_length:.2e}_R{self.thread_radius:.2e}_dx{self.dx:.0f}_YM{self.youngs_modulus:.2e}_Density{self.density:.2e}_Damping{self.damping_constant:.0f}_TF{self.tension_force:.0e}_PF{self.point_force_mag:.0e}{self.TYPE_PF}_k{self.k:.0e}_kt{self.kt:.0e}_fps{self.rendering_fps}'
        name = f"FiberSim_{self.num_horizontal_threads+self.num_vertical_threads}rods_{suffix}" #_multiplevib"
        print(name)

        self.add_threads()

        point_force = np.array([0.0, self.point_force_mag, 0.0])   # N -> kg m/s2 --> 1e3 g 1e3 mm / s2 (1e6)
        node_idx = np.rint(self.n_elem/2.).astype(int)

        if self.SPREAD_PF:
            point_force_spread = np.arange(-2, 3)
        else:
            point_force_spread = np.arange(0, 1)

        stencil = 1/(np.abs(point_force_spread)+1)
        stencil /= np.sum(stencil)

        if self.TYPE_PF=="constant":
            for i in point_force_spread:
                self.simulator.add_forcing_to(self.horizontal_thread[-1]).using(
                    PointForce, node_idx=node_idx+i, point_force=point_force*stencil[i],
                    ramp_up_time=0.25, hold_time=5.0)
        elif self.TYPE_PF=="sinusoidal":
            for i in point_force_spread:
                self.simulator.add_forcing_to(self.horizontal_thread[-1]).using(
                    PointForceSinsusoidal, node_idx=node_idx+i, point_force=point_force*stencil[i],
                    ramp_up_time=0.25, hold_time=5.0)
        elif self.TYPE_PF=="spline":
            sample_time = np.ceil(self.duration).astype(int)
            y_sample = np.random.uniform(-1,1, size=sample_time*self.sample_freq+1)
            y_sample[0] = 0.0

            x_sample = np.linspace(0, sample_time, sample_time*self.sample_freq + 1)
            spline = CubicSpline(x_sample, y_sample)

            for i in point_force_spread:
                self.simulator.add_forcing_to(self.horizontal_thread[-1]).using(
                    PointForceSpline, node_idx=node_idx+i, point_force=point_force*stencil[i],
                    ramp_up_time=1.0, spline=spline)
        else:
            print("Invalid type of point force!!")
            
        self.simulator.finalize()

        current_time = 0.0
        n_steps = np.rint((self.duration ) / self.sim_dt).astype(int)

        horizontal_state = np.zeros((self.num_horizontal_threads, 3, self.n_elem+1))
        vertical_state = np.zeros((self.num_vertical_threads, 3, self.n_elem+1))

        do_step, stages_and_updates = extend_stepper_interface(self.StatefulStepper, self.simulator)

        for i in tqdm(range(n_steps)):
            current_time = do_step(self.StatefulStepper, stages_and_updates, self.simulator, current_time, self.sim_dt)

            for j in range(self.num_horizontal_threads):
                horizontal_state[j, 0:3, ...] = self.horizontal_thread[j].position_collection

            for k in range(self.num_vertical_threads):
                vertical_state[k, 0:3, ...] = self.vertical_thread[k].position_collection

            state = np.concatenate((horizontal_state, vertical_state), axis=0)

            if self.STOP_AT_NAN:
                if _isnan_check(state):
                    print(f"NaN values encountered at {current_time}")
                    stopped_at_nan = True
                    break
                else:
                    stopped_at_nan = False

        if not stopped_at_nan:
            rods_history = self.post_processing_dict_horizontal_thread + self.post_processing_dict_vertical_thread
            time = np.array(rods_history[0]["time"])
            if self.TYPE_PF=="constant":
                force_profile = np.full_like(time, self.point_force_mag)
            elif self.TYPE_PF=="sinusoidal":
                force_profile = np.sin(time*(2 * np.pi)/0.3)*np.sin(time*(2 * np.pi)/1.0)*self.point_force_mag
            elif self.TYPE_PF=="spline":
                force_profile = spline(time)*self.point_force_mag
            else:
                print("Invalid type of point force!!")

            data = ([rods_history] + [force_profile])
            
            with open(f'{name}.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Data saved as Pickle!!")

            df = rods_history[0]
            df = pd.DataFrame(df)
            df.to_csv(f'{name}.csv', index=False)
            print("Data saved as CSV!!")

            if self.VIDEO:
                x_limits = [-self.thread_length/2-5, self.thread_length/2+5]
                y_limits = [-self.thread_length/2-5, self.thread_length/2+5]
                params_str =  f"Young's Modulus = {self.youngs_modulus:.2e}, Point Force = {self.point_force_mag:.0e}, Tension Force = {self.tension_force:.0e}"
                plot_network_video_2D(
                    rods_history,
                    video_name=f"{name}.mp4",
                    fps=self.rendering_fps,
                    step=1,
                    vis2D=False,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    params_str=params_str
                )
                os.remove(f'{name}.pickle')
                print("Video saved!! Pickle file deleted!!")
        else:
            print("Stopped at NaN. Data not saved!")