import os
import time
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
from utils.render.post_processing import plot_network_video, plot_network_video_2D, plot_network_video_2D_less_callback

from tqdm import tqdm

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass

class fiber_simulation():

    def __init__(self, **kwargs):

        self.simulator = BaseSimulator()

        """NETWORK GEOMETRY"""
        self.num_sides_polygon = kwargs.get("num_sides_polygon", 3)
        self.network_origin = kwargs.get("network_origin", np.zeros((3,)))

        """COMMON THREAD PROPERTIES"""
        self.polygon_diameter = kwargs.get("polygon_diameter", 1000e-3 * 1e3) # m --> 1e3 mm (1e3) 
        thread_diameter = kwargs.get("thread_diameter", 4e-3 * 1e3) # m --> 1e3 mm (1e3)
        self.thread_radius = 0.5 * thread_diameter
        self.dx = kwargs.get("dx", 20) # m --> 1e3 mm (1e3)

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

        '''Magnitude and type of external point force'''
        self.point_force_mag = kwargs.get("point_force_mag", -2e5)
        self.SPREAD_PF = kwargs.get("SPREAD_PF", False)
        self.spread = kwargs.get("spread", 2)
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
        self.SAVE = kwargs.get("SAVE", True)
        self.VIDEO = kwargs.get("VIDEO", True)

        self.scaling_type = kwargs.get('scaling_type', "mm_g_s")
        self.loc = kwargs.get('loc', './')
        self.file_type = kwargs.get('file_type', 'pickle')
        self.n_file = kwargs.get('n_file', 0)

    def add_threads(self):
        """RADIAL THREADS"""
        self.radial_threads = [None for i in range(int(self.num_sides_polygon/2))] #int(self.num_sides_polygon/2)
        self.n_elem_radial = np.rint(self.polygon_diameter/self.dx).astype(int)
        print(f"Number of elements in radial threads: {self.n_elem_radial}")
        ang_between_radial_threads = 2 * np.pi / self.num_sides_polygon

        self.rad_rad_connect_idx = np.rint(self.n_elem_radial/2).astype(int)
        rad_to_diag_connect_distance_from_center = 0.5 * self.polygon_diameter * np.cos(2 * np.pi / self.num_sides_polygon)
        self.rad_to_diag_connect_idx_near_start = np.rint((0.5*self.polygon_diameter - rad_to_diag_connect_distance_from_center)/self.dx).astype(int)
        self.rad_to_diag_connect_idx_near_end = np.rint((0.5*self.polygon_diameter + rad_to_diag_connect_distance_from_center)/self.dx).astype(int)
        print(f"Index of connection between radial threads: {self.rad_rad_connect_idx}")
        print(f"Index of connection between radial and diagonal threads: {self.rad_to_diag_connect_idx_near_start}, {self.rad_to_diag_connect_idx_near_end}")

        def _vertices(num_sides, diameter, origin, ang_between_radial_threads):
            vertices = []
            for i in range(num_sides):
                angle = i * ang_between_radial_threads
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                vertex = origin - 0.5 * diameter * direction
                if vertex[0] < 1e-10 and vertex[0] > -1e-10:
                    vertex[0] = 0.0
                if vertex[1] < 1e-10 and vertex[1] > -1e-10:
                    vertex[1] = 0.0
                if vertex[2] < 1e-10 and vertex[2] > -1e-10:
                    vertex[2] = 0.0
                vertices.append(vertex)
            return vertices
        
        vertices = _vertices(self.num_sides_polygon, self.polygon_diameter, self.network_origin, ang_between_radial_threads)
        
        # Radial threads will start from the first half of the vertices
        for i in range(len(self.radial_threads)):
            
            ang = i * ang_between_radial_threads
            radial_thread_direction = np.array([np.cos(ang), np.sin(ang), 0.0])
            radial_thread_start = vertices[i]

            self.radial_threads[i] = CosseratRod.straight_rod(
                n_elements=self.n_elem_radial,
                start=radial_thread_start,
                direction=radial_thread_direction,
                normal=self.normal,
                base_length=self.polygon_diameter,
                base_radius=self.thread_radius,
                density=self.density,
                youngs_modulus=self.youngs_modulus,
            )

            self.simulator.append(self.radial_threads[i])

            self.simulator.dampen(self.radial_threads[i]).using(
                AnalyticalLinearDamper, 
                damping_constant = self.damping_constant,
                time_step = self.sim_dt
            )

            # self.simulator.dampen(self.radial_threads[i]).using(LaplaceDissipationFilter, filter_order = self.filter_order)

            self.simulator.constrain(self.radial_threads[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(0, ), 
                constrained_director_idx=(0, ), 
                translational_constraint_selector=np.array([True, True, True]),
                rotational_constraint_selector=np.array([False, False, False]),
            )

            self.simulator.constrain(self.radial_threads[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(-1, ), 
                constrained_director_idx=(-1, ), 
                translational_constraint_selector=np.array([False, False, True]),
                rotational_constraint_selector=np.array([False, False, False]),
            )

            tension_force_vector = self.tension_force * radial_thread_direction

            self.simulator.add_forcing_to(self.radial_threads[i]).using(
                EndpointForces, -tension_force_vector, tension_force_vector, ramp_up_time=0.25
            )

            if i > 0:
                self.simulator.connect(self.radial_threads[i-1], self.radial_threads[i],
                                       int(self.rad_rad_connect_idx), int(self.rad_rad_connect_idx)).using(FixedJoint,
                                                                                                           k = self.k,
                                                                                                           nu = self.nu,
                                                                                                           kt = self.kt)
                
            

        """DIAGONAL THREADS"""
        self.num_diagonal_threads = self.num_sides_polygon
        self.diagonal_threads = [None for i in range(self.num_diagonal_threads)]
        self.diagonal_length = self.polygon_diameter * np.sin(2*np.pi/self.num_sides_polygon)
        self.n_elem_diagonal = np.rint(self.diagonal_length/self.dx).astype(int)

        diag_to_rad_connect_distance = 0.5 * self.diagonal_length
        self.diag_to_rad_connect_idx = np.rint(diag_to_rad_connect_distance/self.dx).astype(int)

        diag_diag_connect_distance = 0.5 * self.polygon_diameter * np.tan(np.pi/self.num_sides_polygon)
        self.diag_diag_connect_idx_near_start = np.rint((diag_diag_connect_distance)/self.dx).astype(int)
        self.diag_diag_connect_idx_near_end = np.rint((self.diagonal_length - diag_diag_connect_distance)/self.dx).astype(int)

        # Diagonal threads will connect every second vertex of the polygon i.e., vertex 0 to vertex 2, vertex 1 to vertex 3, and so on.

        for i in range(self.num_diagonal_threads):
            ## coordinates of current vertex of the polygon
            vertex_1 = vertices[i]

            ## coordinates of next to next vertex of the polygon
            vertex_2 = vertices[(i + 2) % self.num_sides_polygon]

            diagonal_thread_direction = vertex_2 - vertex_1
            diagonal_thread_direction /= np.linalg.norm(diagonal_thread_direction)

            diagonal_thread_start = vertex_1

            self.diagonal_threads[i] = CosseratRod.straight_rod(
                n_elements=self.n_elem_diagonal,
                start=diagonal_thread_start,
                direction=diagonal_thread_direction,
                normal=self.normal,
                base_length=self.diagonal_length,
                base_radius=self.thread_radius,
                density=self.density,
                youngs_modulus=self.youngs_modulus,
            )

            self.simulator.append(self.diagonal_threads[i])

            self.simulator.dampen(self.diagonal_threads[i]).using(
                AnalyticalLinearDamper, 
                damping_constant = self.damping_constant,
                time_step = self.sim_dt
            )

            # self.simulator.dampen(self.diagonal_threads[i]).using(LaplaceDissipationFilter, filter_order = self.filter_order)

            self.simulator.constrain(self.diagonal_threads[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(0, ), 
                constrained_director_idx=(0, ), 
                translational_constraint_selector=np.array([True, True, True]),
                rotational_constraint_selector=np.array([False, False, False]),
            )

            self.simulator.constrain(self.diagonal_threads[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(-1, ), 
                constrained_director_idx=(-1, ), 
                translational_constraint_selector=np.array([False, False, True]),
                rotational_constraint_selector=np.array([False, False, False]),
            )

            # tension_force_vector = self.tension_force * diagonal_thread_direction

            # self.simulator.add_forcing_to(self.diagonal_threads[i]).using(
            #     EndpointForces, -tension_force_vector, tension_force_vector, ramp_up_time=0.25
            # )

            if i > 1:
                self.simulator.connect(self.diagonal_threads[i-1], self.diagonal_threads[i],
                                       int(self.diag_diag_connect_idx_near_end), int(self.diag_diag_connect_idx_near_start)).using(FixedJoint,
                                                                                                           k = 1e-2*self.k,
                                                                                                           nu = self.nu,
                                                                                                           kt = self.kt)
                
            if i == self.num_diagonal_threads -1:
                self.simulator.connect(self.diagonal_threads[i], self.diagonal_threads[0],
                                       int(self.diag_diag_connect_idx_near_end), int(self.diag_diag_connect_idx_near_start)).using(FixedJoint,
                                                                                                           k = 1e-2*self.k,
                                                                                                           nu = self.nu,
                                                                                                           kt = self.kt)

        """ADDING CONNECTIONS BETWEEN RADIAL AND DIAGONAL THREADS"""
        for i in range(len(self.radial_threads)):
            diag_thread_near_rad_start = (i - 1) % self.num_sides_polygon
            diag_thread_near_rad_end = (diag_thread_near_rad_start + self.num_sides_polygon//2) % self.num_sides_polygon

            self.simulator.connect(self.radial_threads[i], self.diagonal_threads[diag_thread_near_rad_start],
                                   int(self.rad_to_diag_connect_idx_near_start), int(self.diag_to_rad_connect_idx)).using(FixedJoint,
                                                                                                       k = self.k,
                                                                                                       nu = self.nu,
                                                                                                       kt = self.kt)
            self.simulator.connect(self.radial_threads[i], self.diagonal_threads[diag_thread_near_rad_end],
                                   int(self.rad_to_diag_connect_idx_near_end), int(self.diag_to_rad_connect_idx)).using(FixedJoint,
                                                                                                       k = self.k,
                                                                                                       nu = self.nu,
                                                                                                       kt = self.kt)
            
        """ADDING CONNECTIONS AT END POINTS OF RADIAL THREADS AND DIAGONAL THREADS"""
        for i in range(len(self.radial_threads)):
            diag_a = i % self.num_sides_polygon # diagonal starting at node 0 of radial thread i
            diag_b = (i - 2) % self.num_sides_polygon # diagonal ending at node -1 of radial thread i

            self.simulator.connect(self.radial_threads[i], self.diagonal_threads[diag_a],
                                   0, 0).using(FixedJoint,
                                               k = self.k,
                                               nu = self.nu,
                                               kt = self.kt)
            
            self.simulator.connect(self.radial_threads[i], self.diagonal_threads[diag_b],
                                   0, -1).using(FixedJoint,
                                               k = self.k,
                                               nu = self.nu,
                                               kt = self.kt)
            
            diag_c = (i + len(self.radial_threads)) % self.num_sides_polygon # diagonal starting at node -1 of radial thread i
            diag_d = (i + len(self.radial_threads) - 2) % self.num_sides_polygon # diagonal ending at node -1 of radial thread i

            self.simulator.connect(self.radial_threads[i], self.diagonal_threads[diag_c],
                                   -1, 0).using(FixedJoint,
                                               k = self.k,
                                               nu = self.nu,
                                               kt = self.kt)
            
            self.simulator.connect(self.radial_threads[i], self.diagonal_threads[diag_d],
                                   -1, -1).using(FixedJoint,
                                               k = self.k,
                                               nu = self.nu,
                                               kt = self.kt)

        """INITIALIZING CALLBACK"""
        if self.SAVE or self.VIDEO:            
            self.post_processing_dict_radial_thread = []
            for i in range(len(self.radial_threads)):
                post_processing_dict_radial_thread_each = defaultdict(list)
                self.simulator.collect_diagnostics(self.radial_threads[i]).using(
                    NetworkCallBack,
                    step_skip=self.step_skip,
                    callback_params=post_processing_dict_radial_thread_each,
                )

                self.post_processing_dict_radial_thread.append(post_processing_dict_radial_thread_each)

            self.post_processing_dict_diagonal_thread = []
            for i in range(self.num_diagonal_threads):
                post_processing_dict_diagonal_thread_each = defaultdict(list)
                self.simulator.collect_diagnostics(self.diagonal_threads[i]).using(
                    NetworkCallBack,
                    step_skip=self.step_skip,
                    callback_params=post_processing_dict_diagonal_thread_each,
                )

                self.post_processing_dict_diagonal_thread.append(post_processing_dict_diagonal_thread_each)

    # def get_node_idx(self):
    #     if self.num_horizontal_threads == 1 and self.num_vertical_threads == 0:
    #         vib_thread_idx_list = np.array([0])
    #         node_idx_list = np.array([self.n_elem//2])
    #     elif self.num_horizontal_threads == 2:
    #         vib_thread_idx_list = np.array([0])
    #         node_idx_list = np.array([self.n_elem//2])
    #     elif self.num_horizontal_threads == 3:
    #         vib_thread_idx_list = np.array([self.num_horizontal_threads//2])
    #         node_idx_list = np.array([(self.hor_connect_idx[0]+self.hor_connect_idx[1])//2]).astype(int)
    #     else:
    #     #     # Number of threads to be stimulated
    #     #     num_vib_threads = np.floor(self.num_horizontal_threads*0.75).astype(int)

    #     #     # Randomly select the threads to be stimulated
    #     #     vib_thread_idx_list = np.random.choice(self.num_horizontal_threads, num_vib_threads, replace=False) # replace=True will enable repeated selection

    #     #     # Randomly select the nodes to be stimulated
    #     #     valid_nodes = [i for i in range(self.n_elem+1) if all(abs(i-idx) >= self.spread for idx in self.hor_connect_idx) and min(self.hor_connect_idx) <= i <= max(self.hor_connect_idx)]
    #     #     node_idx_list = np.random.choice(valid_nodes, num_vib_threads, replace=True)
            
    #         vib_thread_idx_list = np.array([self.num_horizontal_threads//2])
    #         node_idx_list = np.array([self.n_elem//2])

    #     return vib_thread_idx_list, node_idx_list
    
    def save_data(self, data, name):
        if self.file_type == "pickle":
            with open(f'{self.loc}{name}.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Data saved as Pickle!!")
        elif self.file_type == "npz":
            rods_history = data
            np.savez(f'{self.loc}{name}.npz', rods_history=rods_history)
            print("Data saved as NPZ!!")
        else:
            print("Invalid file type!!")

    def launch_sim(self):

        if self.scaling_type == "mm_g_s":
            time_scale = 1
            length_scale = 1e3
            mass_scale = 1e3
            force_scale = 1e6
            density_scale = 1e-6
            modulus_scale = 1
        elif self.scaling_type == "mm_mg_ms":
            time_scale = 1e3
            length_scale = 1e3
            mass_scale = 1e6
            force_scale = 1e3
            density_scale = 1e-3
            modulus_scale = 1e-3
        else:
            raise NotImplementedError ("This unit scaling has not been implemented")

        # suffix = f'{self.duration/time_scale:.0f}sec_L{self.polygon_diameter/length_scale:.2e}m_R{self.thread_radius/length_scale:.2e}m_dx{self.dx:.0f}mm_YM{self.youngs_modulus/modulus_scale:.2e}Pa_Density{self.density/density_scale:.2e}kgmm-3_Damping{self.damping_constant:.0f}_TF{self.tension_force/force_scale:.0e}N_PF{self.point_force_mag/force_scale:.0e}N{self.TYPE_PF}_k{self.k:.0e}_kt{self.kt:.0e}_fps{self.rendering_fps}_stepskip{self.step_skip}'
        # # name = f"{self.scaling_type}_FiberSim_{self.num_horizontal_threads+self.num_vertical_threads}rods_{suffix}"
        # name = f"diffconstraints_{self.scaling_type}_FiberSim_{self.num_sides_polygon}rods_{suffix}_{self.n_file}"
        name = f"Polygon{self.num_sides_polygon}_PF{self.point_force_mag/force_scale:.0e}N{self.TYPE_PF}_{self.sample_freq}Hz_{self.n_file}_changedBC"
        print(name)

        self.add_threads()

        point_force = np.array([0.0, self.point_force_mag, 0.0])   # N -> kg m/s2 --> 1e3 g 1e3 mm / s2 (1e6) --> 1e6 mg 1e3 mm / 1e6 ms2 (1e-3)

        if self.SPREAD_PF:
            point_force_spread = np.arange(-self.spread, self.spread+1)
        else:
            point_force_spread = np.arange(0, 1)

        stencil = 1/(np.abs(point_force_spread)+1)
        stencil /= np.sum(stencil)

        # vib_thread_idx_list, node_idx_list = self.get_node_idx()
        vib_thread_idx_list = [0]
        # node_idx = np.rint((self.rad_to_diag_connect_idx_near_start + self.rad_rad_connect_idx)/2).astype(int)
        # node_idx = np.rint((self.rad_to_diag_connect_idx_near_end + self.n_elem_radial + 1)/2).astype(int)
        node_idx = np.rint((self.rad_rad_connect_idx + self.rad_to_diag_connect_idx_near_end)/2).astype(int)
        node_idx_list = [node_idx]

        if self.TYPE_PF=="none":
            pass
        elif self.TYPE_PF=="constant":
            ramp_up_time = 0.25 * time_scale
            hold_time = 5.0 * time_scale
            for j in range(len(vib_thread_idx_list)):
                node_idx = node_idx_list[j]
                vib_thread = self.horizontal_thread[j]
                for i in point_force_spread:
                    self.simulator.add_forcing_to(vib_thread).using(
                        PointForce, node_idx=node_idx+i, point_force=point_force*stencil[i],
                        ramp_up_time=ramp_up_time, hold_time=hold_time)
        elif self.TYPE_PF=="sinusoidal":
            ramp_up_time = 0.25 * time_scale
            hold_time = 5.0 * time_scale
            for j in range(len(vib_thread_idx_list)):
                node_idx = node_idx_list[j]
                vib_thread = self.horizontal_thread[j]
                for i in point_force_spread:
                    self.simulator.add_forcing_to(vib_thread).using(
                        PointForceSinsusoidal, node_idx=node_idx+i, point_force=point_force*stencil[i],
                        ramp_up_time=ramp_up_time, hold_time=hold_time)
        elif self.TYPE_PF=="spline":
            ramp_up_time = 1.0 * time_scale
            seed_value = 1234 #int(time.time()) % (2**32-1)
            np.random.seed(seed_value)

            sample_time = np.ceil(self.duration).astype(int)
            x_sample = np.linspace(0, sample_time, sample_time*self.sample_freq + 1)

            spline_list = []

            for j in range(len(vib_thread_idx_list)):
                y_sample = np.random.uniform(-1,1, size=sample_time*self.sample_freq+1)
                y_sample[0] = 0.0    
                spline = CubicSpline(x_sample, y_sample)

                node_idx = node_idx_list[j]
                vib_thread_idx = vib_thread_idx_list[j]
                vib_thread = self.radial_threads[vib_thread_idx]
                
                for i in point_force_spread:
                    self.simulator.add_forcing_to(vib_thread).using(
                        PointForceSpline, node_idx=node_idx+i, point_force=point_force*stencil[i],
                        ramp_up_time=ramp_up_time, spline=spline)
                    
                spline_list.append(spline)
        else:
            raise NotImplementedError ("Invalid type of point force!!")
            
        self.simulator.finalize()

        current_time = 0.0
        n_steps = np.rint((self.duration ) / self.sim_dt).astype(int)

        radial_state = np.zeros((len(self.radial_threads), 3, self.n_elem_radial+1))
        diagonal_state = np.zeros((self.num_diagonal_threads, 3, self.n_elem_diagonal+1))

        do_step, stages_and_updates = extend_stepper_interface(self.StatefulStepper, self.simulator)

        for i in tqdm(range(n_steps)):
            current_time = do_step(self.StatefulStepper, stages_and_updates, self.simulator, current_time, self.sim_dt)

            for j in range(len(self.radial_threads)):
                radial_state[j, 0:3, ...] = self.radial_threads[j].position_collection

            for k in range(self.num_diagonal_threads):
                diagonal_state[k, 0:3, ...] = self.diagonal_threads[k].position_collection

            # state = np.concatenate((radial_state, vertical_state), axis=0)
            state = radial_state

            if self.STOP_AT_NAN:
                if _isnan_check(state):
                    print(f"NaN values encountered at {current_time}")
                    stopped_at_nan = True
                    break
                else:
                    stopped_at_nan = False
            else:
                stopped_at_nan = False

        if not stopped_at_nan:
            rods_history = self.post_processing_dict_radial_thread + self.post_processing_dict_diagonal_thread
            time_array = np.array(rods_history[0]["time"])
            # if self.TYPE_PF=="constant":
            #     force_profile = np.full_like(time_array, self.point_force_mag)
            # elif self.TYPE_PF=="sinusoidal":
            #     force_profile = np.sin(time_array*(2 * np.pi)/0.3)*np.sin(time_array*(2 * np.pi)/1.0)*self.point_force_mag
            # elif self.TYPE_PF=="spline":
            #     force_profile = [spline_i(time_array)*self.point_force_mag for spline_i in spline_list]
            # elif self.TYPE_PF=="varying_sine":
            #     force_profile = [np.sin(2*np.pi*spline_i(time_array)*time_array)*self.point_force_mag for spline_i in spline_list]
            #     print(len(force_profile))
            # else:
            #     print("Invalid type of point force!!")

            data = ([rods_history])# + [force_profile] + [seed_value])
            
            if self.SAVE:
                self.save_data(data, name)

            if self.VIDEO:
                x_limits = [-self.polygon_diameter/2-5, self.polygon_diameter/2+5]
                y_limits = [-self.polygon_diameter/2-5, self.polygon_diameter/2+5]
                params_str =  f"Young's Modulus = {self.youngs_modulus/modulus_scale:.2e}Pa, Point Force = {self.point_force_mag/force_scale:.0e}N, Tension Force = {self.tension_force/force_scale:.0e}N"
                plot_network_video_2D_less_callback(
                    rods_history,
                    video_name=f"{self.loc}{name}.mp4",
                    fps=self.rendering_fps,
                    step=1,
                    vis2D=False,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    params_str=params_str
                )
                # os.remove(f'{name}.{self.file_type}')
                print("Video saved!! Pickle file deleted!!")
        else:
            print("Stopped at NaN. Data not saved!")