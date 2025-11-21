import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import multiprocessing

from elastica import *
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica.modules.damping import Damping

from gym_fiber.utils.custom_elastica.pointforce import PointForce, PointForceSinsusoidal, PointForceSpline
from gym_fiber.utils.custom_elastica.networkcallback import NetworkCallBack
from gym_fiber.utils.render.post_processing import plot_network_video, plot_network_video_2D

from tqdm import tqdm

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass

class fiber_simulation():

    def __init__(self, **kwargs):

        self.simulator = BaseSimulator()

        """NETWORK GEOMETRY"""
        self.num_horizontal_threads = kwargs.get("num_horizontal_threads", 2)
        self.num_vertical_threads = kwargs.get("num_vertical_threads", self.num_horizontal_threads)
        self.network_origin = kwargs.get("network_origin", np.zeros((3,))) #np.array([500, 200, 0.0]) #

        """COMMON THREAD PROPERTIES"""
        self.thread_length = kwargs.get("thread_length", 1000e-3 * 1e3) # m --> 1e3 mm (1e3)
        thread_diameter = kwargs.get("thread_diameter", 4e-3 * 1e3) # m --> 1e3 mm (1e3)
        self.thread_radius = 0.5 * thread_diameter
        self.dx = kwargs.get("dx", 20) # um
        self.n_elem = np.rint(self.thread_length/self.dx).astype(int)
        # print(self.n_elem)

        '''Young's modulus'''
        self.youngs_modulus = kwargs.get("youngs_modulus", 230e6) # Pa (N/m2) => kg / m / s2 --> 1e3 g / 1e3 mm / s2 (1.0)
        # poisson_ratio = 0.45

        # '''Shear modulus'''
        # shear_modulus = 0.5 * self.youngs_modulus / (poisson_ratio + 1.0) #0.98 * 1e6 #
        # poisson_ratio = 1 - 0.5 * self.youngs_modulus/shear_modulus

        '''Density'''
        self.density = kwargs.get("density", 1000 * 1e-6) # kg/m3 --> 1e3 g / 1e9 mm3 (1e-6)

        '''Damping Constant'''
        self.damping_constant = kwargs.get("damping_constant", 5)
        self.filter_order = kwargs.get("filer_order", 6)

        '''Tension Force holding the threads'''
        self.tf = kwargs.get("tf", 1e4)
        self.horizontal_tension_force = np.array([self.tf, 0.0, 0.0]) #1e3 --> 1 uN
        self.vertical_tension_force = np.array([0.0, self.tf, 0.0]) #1e3 --> 1 uN

        '''Magnitude of external point force'''
        self.point_force_mag = kwargs.get("point_force_mag", -2e5)

        '''Thread to be vibrated'''
        self.vib_thread_idx = kwargs.get("vib_thread_idx", [int(self.num_horizontal_threads/2)])

        """CONNECTION PARAMETERS"""
        self.k = kwargs.get("k", 1e9)
        self.nu = kwargs.get("nu", 0.0)
        self.kt = kwargs.get("kt", 1e9)

        """SIMULATION PARAMETERS"""
        '''Total Simulation time'''
        self.max_episode_final_time = kwargs.get("max_episode_final_time", 10)

        '''Physical simulation timestep'''
        self.sim_dt = 5.0e-6 # ms

        '''Type of timestepper'''
        self.StatefulStepper = PositionVerlet()

        '''FPS for video'''
        self.rendering_fps = kwargs.get("rendering_fps", 1000)
        self.step_skip = np.rint(1.0 / (self.rendering_fps * self.sim_dt)).astype(int)

        self.normal = np.array([0.0, 0.0, 1.0])
        self.x_direction = np.array([1.0, 0.0, 0.0])
        self.y_direction = np.array([0.0, 1.0, 0.0])

        self.sample_freq = kwargs.get("sample_freq", 5)

        self.STOP_AT_NAN = False

    def add_threads(self):
        """HORIZONTAL THREADS"""
        self.horizontal_thread = [None for i in range(self.num_horizontal_threads)]

        # horizontal_thread_y_pos[0] = 100 #-100
        # horizontal_thread_y_pos[-1] = 400 #200
        self.vert_connect_idx = np.linspace(0, self.n_elem+1, self.num_horizontal_threads+2)
        self.vert_connect_idx = self.vert_connect_idx[1:-1]
        for i in range(len(self.vert_connect_idx)):
            self.vert_connect_idx[i] = int(self.vert_connect_idx[i])
        horizontal_thread_y_pos = self.vert_connect_idx * self.dx - self.thread_length/2 + self.network_origin[1]
        # print(horizontal_thread_y_pos)
        # self.vert_connect_idx[0] = int(self.n_elem * 0.4)
        # self.vert_connect_idx[-1] = int(self.n_elem * 0.7)
        # print(self.vert_connect_idx)

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
                # shear_modulus=shear_modulus
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
                rotational_constraint_selector=np.array([True, True, True]),
            )

            self.simulator.constrain(self.horizontal_thread[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(-1, ), 
                constrained_director_idx=(-1, ), 
                translational_constraint_selector=np.array([True, True, True]),
                rotational_constraint_selector=np.array([True, True, True]),
            )

            self.simulator.add_forcing_to(self.horizontal_thread[i]).using(
                EndpointForces, -self.horizontal_tension_force, self.horizontal_tension_force, ramp_up_time=0.25
            )

        """VERTICAL THREADS"""
        self.vertical_thread = [None for i in range(self.num_vertical_threads)]
        # vertical_thread_x_pos[0] = 300 #-200
        # vertical_thread_x_pos[-1] = 800 #300
        self.hor_connect_idx = np.linspace(0, self.n_elem+1, self.num_vertical_threads+2)
        self.hor_connect_idx = self.hor_connect_idx[1:-1]
        for i in range(len(self.hor_connect_idx)):
            self.hor_connect_idx[i] = int(self.hor_connect_idx[i])
        vertical_thread_x_pos = self.hor_connect_idx * self.dx - self.thread_length/2 + self.network_origin[0]
        # self.hor_connect_idx[0] = int(self.n_elem * 0.3)
        # self.hor_connect_idx[-1] = int(self.n_elem * 0.8)
        # print(self.hor_connect_idx)

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
                # shear_modulus=shear_modulus
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
                rotational_constraint_selector=np.array([True, True, True])
            )

            self.simulator.constrain(self.vertical_thread[i]).using(
                GeneralConstraint, 
                constrained_position_idx=(-1, ), 
                constrained_director_idx=(-1, ),
                translational_constraint_selector=np.array([True, True, True]), 
                rotational_constraint_selector=np.array([True, True, True])
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

        suffix = f'sweep_{self.max_episode_final_time:.0f}sec_{self.sample_freq:.0f}hz_d{self.damping_constant:.0f}_tension{self.tf:.0e}_pointforce{self.point_force_mag:.0e}_k{self.k:.0e}_kt{self.kt:.0e}_fps{self.rendering_fps}'
        name = f"{self.num_horizontal_threads+self.num_vertical_threads}rods_{suffix}_multiplevib"
        print(name)

        self.add_threads()
        
        sample_time = np.ceil(self.max_episode_final_time).astype(int)
        y_sample = np.random.uniform(-1,1, size=sample_time*self.sample_freq+1)
        y_sample[0] = 0.0

        x_sample = np.linspace(0, sample_time, sample_time*self.sample_freq + 1)
        spline = CubicSpline(x_sample, y_sample)

        point_force = np.array([0.0, self.point_force_mag, 0.0])   # N -> kg m/s2 --> 1e3 g 1e3 mm / s2 (1e6)
        
        point_force_spread = np.arange(-2, 3)
        stencil = 1/(np.abs(point_force_spread)+1)
        stencil /= np.sum(stencil)

        for j in range(len(self.vib_thread_idx)):
            # 8 rods
            # if j == 1:
            #     node_idx = np.rint(self.n_elem/2.0).astype(int)
            # else:
            #     node_idx = np.rint((self.hor_connect_idx[1]+self.hor_connect_idx[0])/2.0).astype(int)
            # 16 rods
            if j == 2:
                node_idx = np.rint(self.n_elem/2.0).astype(int)
            elif j == 1:
                node_idx = np.rint((self.hor_connect_idx[2]+self.hor_connect_idx[1])/2.0).astype(int)
            elif j == 0:
                node_idx = np.rint((self.hor_connect_idx[5]+self.hor_connect_idx[6])/2.0).astype(int)
            else:
                node_idx = np.rint((self.hor_connect_idx[1]+self.hor_connect_idx[0])/2.0).astype(int)
            
            for i in point_force_spread:
                self.simulator.add_forcing_to(self.horizontal_thread[self.vib_thread_idx[j]]).using(
                    PointForceSpline, node_idx=node_idx+i, point_force=point_force*stencil[i],
                    ramp_up_time=1.0, spline=spline)
            
        self.simulator.finalize()

        current_time = 0.0
        n_steps = np.rint((self.max_episode_final_time ) / self.sim_dt).astype(int)

        horizontal_bending_energy = np.zeros((self.num_horizontal_threads, n_steps))
        horizontal_shear_energy = np.zeros((self.num_horizontal_threads, n_steps))
        horizontal_translational_energy = np.zeros((self.num_horizontal_threads, n_steps))
        horizontal_rotational_energy = np.zeros((self.num_horizontal_threads, n_steps))
        horizontal_state = np.zeros((self.num_horizontal_threads, 3, self.n_elem+1))

        vertical_bending_energy = np.zeros((self.num_vertical_threads, n_steps))
        vertical_shear_energy = np.zeros((self.num_vertical_threads, n_steps))
        vertical_translational_energy = np.zeros((self.num_vertical_threads, n_steps))
        vertical_rotational_energy = np.zeros((self.num_vertical_threads, n_steps))
        vertical_state = np.zeros((self.num_vertical_threads, 3, self.n_elem+1))

        do_step, stages_and_updates = extend_stepper_interface(self.StatefulStepper, self.simulator)

        for i in tqdm(range(n_steps)):
            current_time = do_step(self.StatefulStepper, stages_and_updates, self.simulator, current_time, self.sim_dt)

            for j in range(self.num_horizontal_threads):
                # horizontal_bending_energy[j, i] = CosseratRod.compute_bending_energy(self.horizontal_thread[j])
                # horizontal_shear_energy[j, i] = CosseratRod.compute_shear_energy(self.horizontal_thread[j])
                # horizontal_translational_energy[j, i] = CosseratRod.compute_translational_energy(self.horizontal_thread[j])
                # horizontal_rotational_energy[j, i] = CosseratRod.compute_rotational_energy(self.horizontal_thread[j])
                horizontal_state[j, 0:3, ...] = self.horizontal_thread[j].position_collection

            for k in range(self.num_vertical_threads):
                # vertical_bending_energy[k, i] = CosseratRod.compute_bending_energy(self.vertical_thread[k])
                # vertical_shear_energy[k, i] = CosseratRod.compute_shear_energy(self.vertical_thread[k])
                # vertical_translational_energy[k, i] = CosseratRod.compute_translational_energy(self.vertical_thread[k])
                # vertical_rotational_energy[k, i] = CosseratRod.compute_rotational_energy(self.vertical_thread[k])
                vertical_state[k, 0:3, ...] = self.vertical_thread[k].position_collection

            state = np.concatenate((horizontal_state, vertical_state), axis=0)

            if self.STOP_AT_NAN:
                if _isnan_check(state):
                    print(f"NaN values encountered at {current_time}")
                    break

            
        rods_history = self.post_processing_dict_horizontal_thread + self.post_processing_dict_vertical_thread
        time = np.array(rods_history[0]["time"])
        force_profile = spline(time)

        data = ([rods_history] + [force_profile])
        
        with open(f'processing_data/{name}.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Data saved!!")

        x_limits = [-self.thread_length/2-5, self.thread_length/2+5]
        y_limits = [-self.thread_length/2-5, self.thread_length/2+5]
        # x_limits = [-self.thread_length*0.1, self.thread_length*1.1]
        # y_limits = [-self.thread_length*0.4, self.thread_length*0.8]
        params_str =  f"Point Force = {self.point_force_mag:.0e}, k = {self.k:.0e}, kt = {self.kt:.0e}"
        plot_network_video_2D(
            rods_history,
            video_name=f"videos/{name}.mp4",
            fps=self.rendering_fps,
            step=1,
            vis2D=False,
            x_limits=x_limits,
            y_limits=y_limits,
            params_str=params_str
        )
        print("Video saved!!")

def wrapper_launcher(params):
    num_horizontal_threads, num_vertical_threads, network_origin, thread_length, thread_diameter, dx, damping_constant, k, nu, kt, tf, point_force_mag, sample_freq, max_episode_final_time, rendering_fps, vib_thread_idx = params
    print(params)
    try:
        sim = fiber_simulation(
            num_horizontal_threads=num_horizontal_threads, 
            num_vertical_threads=num_vertical_threads, 
            network_origin=network_origin, 
            thread_length=thread_length, 
            thread_diameter=thread_diameter, 
            dx=dx,
            damping_constant=damping_constant, 
            k=k,
            nu=nu,
            kt=kt,
            tf=tf,
            point_force_mag = point_force_mag, 
            sample_freq=sample_freq, 
            max_episode_final_time=max_episode_final_time, 
            rendering_fps=rendering_fps,
            vib_thread_idx=vib_thread_idx
            )
        sim.launch_sim()
    except:
        print('Something failed!', params)
    return

def mp_handler():
    p = multiprocessing.Pool(1)
    p.map(wrapper_launcher, params_list)

if __name__ == '__main__':
    num_horizontal_threads = 8
    num_vertical_threads = num_horizontal_threads
    network_origin = np.zeros((3,))

    thread_length = 1000e-3 * 1e3
    thread_diameter = 4e-3 * 1e3
    dx = 20

    damping_constant = 10

    k = 1e9
    nu = 0.0
    kt = 1e9

    tf = 1e4

    point_force_mag = -3e6

    sample_freq = 5
    max_episode_final_time = 10
    rendering_fps = 250

    vib_thread_idx = [int(num_horizontal_threads/2)-2, int(num_horizontal_threads/2)-1, int(num_horizontal_threads/2), int(num_horizontal_threads/2)+1]
    params_list = []
    params_list.append([
        num_horizontal_threads, 
        num_vertical_threads, 
        network_origin, 
        thread_length, 
        thread_diameter, 
        dx, 
        damping_constant,
        k,
        nu,
        kt, 
        tf,
        point_force_mag, 
        sample_freq, 
        max_episode_final_time, 
        rendering_fps,
        vib_thread_idx
        ])

    ''' Launching the simulation'''
    mp_handler()
    # wrapper_launcher(params=params_list)
    # sim = fiber_simulation(damping_constant=damping_constant, tf=tf, sample_freq=sample_freq, max_episode_final_time=max_episode_final_time, rendering_fps=rendering_fps)
    # sim.launch_sim()