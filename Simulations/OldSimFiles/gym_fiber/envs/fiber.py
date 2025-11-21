from collections import defaultdict
import time
import copy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from typing import Optional, Sequence
import gym
from gym import core
from gym import error, spaces, utils
from gym.utils import seeding

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *
from elastica.external_forces import GravityForces
from elastica.modules.damping import Damping

# from gym_fiber.utils.web_design import web_design

from gym_fiber import RENDERER_CONFIG
from gym_fiber.config import RendererType
from gym_fiber.envs.build import build_arm
from gym_fiber.utils.custom_elastica.networkcallback import (
    NetworkCallBack,
)
from gym_fiber.utils.render.post_processing import plot_video_with_sphere, plot_video_with_sphere_2D
# from gym_fiber.utils.render.matplotlib_video import save_video
from gym_fiber.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)

from gym_fiber.utils.custom_elastica.muscle_torque import (
    MuscleTorquesWithVaryingBetaSplines,
)



# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass


# TODO: generalize this as a Class for online trajectory generation.
# TODO: remove np.random: use fixed generator from seed value. (determinism)


class FiberEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, game_mode:int=1):

        '''Web geometry'''
        # The number of radial threads should be the number of lines starting from origin and extending till the web boundary
        # Thus for a hexagonal web the number of radial threads will be 6
        # Stick with even number of radial threads for now
        self.num_rad_threads = 6
        self.num_boundary_threads = self.num_rad_threads
        self.web_origin = np.zeros((3,))
        self.web_radius = 1000
        self.web_plane = 'xy'
        self.boundary_length = 1000*self.num_boundary_threads
        # If web_radius reduces by a factor of n then reduce the spiral_offset and spiral_param_a by the same factor n to keep the geometry constant
        self.spiral_offset = 100 
        self.spiral_param_a = 22
        self.spiral_n_turns = 6
        
        '''number of elements in each type of thread'''
        self.dia_n_elem = 100 # This number indicates the number of elements in the diameter of the web. Half of these + 1 will be the number of elements in each side of the polygon.
        self.spiral_n_elem_first_half = 50
        self.spiral_n_elem_second_half = 100
        self.spiral_n_elem = (self.spiral_n_elem_first_half + self.spiral_n_elem_second_half) * int(self.spiral_n_turns/2) * self.num_rad_threads

        '''Physical simulation timestep in seconds'''
        self.sim_dt = 1.0e-5

        '''number of timesteps to be taken before updating the RL algorithm'''
        self.RL_update_interval = 0.01  # This is 100 updates per second
        self.num_steps_per_update = np.rint(
            self.RL_update_interval / self.sim_dt
        ).astype(int)

        '''Young's modulus in MPa (scaled SI)'''
        self.youngs_modulus = 10 * 1e9
        # poisson_ratio = 0.45

        '''Shear modulus in MPa (scaled SI)'''
        self.shear_modulus = 0.98 * 1e6 #0.5 * self.youngs_modulus / (poisson_ratio + 1.0)
        poisson_ratio = 1 - 0.5 * self.youngs_modulus/self.shear_modulus

        self.torque_scale = 20

        '''Simulation time in seconds (scaled SI)'''
        self.max_episode_final_time = 10

        '''Defining parameters of the each silk thread in mm (scaled SI)'''
        self.base_length = 1000                 ### DONT CHANGE
        self.radius = 50                        ### DONT CHANGE

        self.COLLECT_DATA_FOR_POSTPROCESSING = True
        self.SAVE_VIDEO = True

        self.number_of_control_points = 4       ### DONT CHANGE
        self.number_of_observation_segments = self.number_of_control_points

        self.rendering_fps = 200
        self.step_skip = np.rint(1.0 / (self.rendering_fps * self.sim_dt)).astype(int)

        self.max_rate_of_change_of_activation = np.infty
        self.target_v_scale = 0.001

        self.mode = game_mode
        self.target_location = np.array([750, 1000, -250])

        self.StatefulStepper = PositionVerlet()

        # normal_14 and/or binormal_14 direction_14 activation (3D)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.number_of_control_points,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.number_of_observation_segments * 2 + 6,),
            dtype=np.float32,
        )

        # Determinism
        self.seed()

        # Rendering-related
        self.viewer = None
        self.renderer = None

        # Store observations and reward
        self.traj = pd.DataFrame({'Agent Steps':[], 'X pos':[], 'Y pos':[], 'Z pos':[],
                                  'Distance from Goal':[], 'Reward':[]})
        self.writeReward = pd.DataFrame({'Reward':[]})
        self.writeEpisodeReward = pd.DataFrame({'Episode Reward':[]})
        self.store_agent_steps = 0
        self.store_reward = 0
        self.episode_reward = 0
        self.agent_steps = 0
        #self.reset()

    def seed(self, seed=None):
        # Deprecated in new gym
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """
        avg_kappa_1 = np.zeros(self.number_of_observation_segments)
        # self.dia_n_elem-1 is number of curvature values due to veroni regions
        avg_length = int((self.dia_n_elem - 1) / (self.number_of_observation_segments))

        for i in range(0, self.number_of_observation_segments - 1):
            lower = np.rint(avg_length * (i)).astype(int)
            upper = np.rint(avg_length * (i + 1)).astype(int)
            avg_kappa_1[i] = self.radial_thread[0].kappa[0, lower:upper].mean()

        lower = np.rint(avg_length * (self.number_of_observation_segments - 1)).astype(int)
        avg_kappa_1[-1] = self.radial_thread[0].kappa[0, lower:].mean()

        avg_kappa_2 = np.zeros(self.number_of_observation_segments)
        for i in range(0, self.number_of_observation_segments - 1):
            lower = np.rint(avg_length * (i)).astype(int)
            upper = np.rint(avg_length * (i + 1)).astype(int)
            avg_kappa_2[i] = self.radial_thread[0].kappa[1, lower:upper].mean()

        lower = np.rint(avg_length * (self.number_of_observation_segments - 1)).astype(int)
        avg_kappa_2[-1] = self.radial_thread[0].kappa[1, lower:].mean()

        state = np.concatenate(
            (
                # rod curvature information
                avg_kappa_1
                * self.base_length
                / (
                    2 * np.pi
                ),  # normal_14ize by the curvature of if the rod is a perfect ring.
                avg_kappa_2
                * self.base_length
                / (
                    2 * np.pi
                ),  # normal_14ize by the curvature of if the rod is a perfect ring.
                # arm tip location
                self.radial_thread[0].position_collection[..., -1] / self.base_length,
                self.radial_thread[0].position_collection[..., -1] / self.base_length,
                # target location
                self.wsol[self.tick] / 1000,  # convert back to meters (~[-1,1])
            )
        )

        return np.array(state, dtype=np.float32)

    def step(self, action):
        # set binormal_14 activations to 0 if solving 2D case
        self.spline_points_func_array_normal_14_dir[:] = action[
            : self.number_of_control_points
        ]
        # print(self.spline_points_func_array_normal_14_dir)
        self.spline_points_func_array_binormal_14_dir[:] = action[
            self.number_of_control_points :
        ]
        # print(self.spline_points_func_array_binormal_14_dir)

        # for _ in range(int(self.num_steps_per_update)):
        #     # print('step:', i)
        #     self.time_tracker = self.do_step(
        #         self.StatefulStepper,
        #         self.stages_and_updates,
        #         self.simulator,
        #         self.time_tracker,
        #         self.sim_dt,
        #     )
        #     self.tick += 1
        #     self.sphere.position_collection[..., 0] = self.wsol[self.tick]
        #     self.sphere.velocity_collection[..., 0] = self.velocity_sphere[self.tick]

        self.time_tracker = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            self.time_tracker,
            self.sim_dt,
        )

        """ Reward Engineering """
        # tip_to_target = (
        #     self.sphere.position_collection[..., 0]
        #     - self.shearable_rod.position_collection[..., -1]
        # ) / 1000
        # reward_dist = -np.square(np.linalg.norm(tip_to_target))
        reward = 1.0

        # observe current state: current as sensed signal
        state = self.get_state()

        # update storing variables
        self.store_agent_steps += 1
        # self.agent_steps += 1
        self.store_reward = reward
        self.writeReward = pd.concat([self.writeReward, pd.DataFrame({'Reward':[reward]})], ignore_index=True)
        # self.writeReward.to_csv('no_policy/Damping 1000/reward_no_policy_RL001_TS10_D1000.csv', index=False)
        self.episode_reward += reward
        
        # if (self.store_agent_steps%100 == 0):
        #     print(f"{self.store_agent_steps} steps done")

        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(state)
        if invalid_values_condition == True:
            reward = -100
            state = np.nan_to_num(self.get_state())

            self.episode_reward = 0

            done = True
            print("Episode blew up. Maybe try a smaller dt?")

        if self.tick * self.sim_dt >= self.max_episode_final_time:

            self.episode_reward = 0

            done = True
            print("Episode has reached max time")

        self.store_reward += reward

        self._target = self.wsol[self.tick]
        # print(self._target)
        return state, reward, done, {"ctime": self.time_tracker}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        # super().reset(seed=seed)
        self.simulator = BaseSimulator()
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None

        ###--------------ADD WEB TO SIMULATION--------------###
        if self.web_plane == 'xy':
            normal = np.array([0.0, 0.0, 1.0])
        else:
            print("Plane of the Spider Web is not XY. Add a new configuration or define the plane")
        x_direction = np.array([1.0, 0.0, 0.0])
        y_direction = np.array([0.0, 1.0, 0.0])

        angle_between_threads = 2 * np.pi / self.num_rad_threads

        # Set the thread properties after defining rods

        damping = self.youngs_modulus * 1e-3
        density = 1000 * 1e-6
        radius_tip = self.radius  # radius of the arm at the tip
        radius_base = radius_tip  # radius of the arm at the base

        radius_along_diameter = np.linspace(radius_tip, radius_tip, self.dia_n_elem)
        dx_dia = 2*self.web_radius/self.dia_n_elem
        damping_constant_rad = damping/(density * dx_dia * np.pi * self.radius**2)

        self.radial_thread = [None for i in range(self.num_rad_threads)]

        for i in range(int(self.num_rad_threads/2)):
            ang = i * angle_between_threads
            direction = np.array([np.cos(ang), np.sin(ang), 0.0])
            start = self.web_origin - self.web_radius * direction
            self.radial_thread[i] = CosseratRod.straight_rod(
                self.dia_n_elem,
                start,
                direction,
                normal,
                2*self.web_radius,
                base_radius=radius_along_diameter,
                density=density,
                # nu=damping,
                youngs_modulus=self.youngs_modulus,
                shear_modulus=self.shear_modulus,
            )

            self.simulator.append(
                self.radial_thread[i]
            )  # Now rod is ready for simulation, append rod to simulation

            self.simulator.dampen(self.radial_thread[i]).using(
                AnalyticalLinearDamper, 
                damping_constant = damping_constant_rad,
                time_step = self.sim_dt
            )

            self.simulator.constrain(self.radial_thread[i]).using(
                FixedConstraint, constrained_position_idx=(0,-1), constrained_director_idx=(0,-1)
            )

            if i > 0:
                self.simulator.connect(self.radial_thread[i-1], self.radial_thread[i], 
                                       int(self.dia_n_elem/2), int(self.dia_n_elem/2)).using(FixedJoint,
                                                                                             k=1e5,
                                                                                             nu=100,
                                                                                             kt=2e3,
                                                                                             nut=0.0)
            # self.simulator.add_forcing_to(self.shearable_rod).using(GravityForces, acc_gravity=np.array([0.0, 9.81*1e3, 0.0]))

        rad_n_elem = int(self.dia_n_elem/2)
        boundary_n_elem = rad_n_elem*self.num_boundary_threads
        boundary_start = self.web_radius*x_direction
        radius_along_boundary = np.linspace(radius_tip, radius_tip, boundary_n_elem)
        dx_bound = self.boundary_length/boundary_n_elem
        damping_constant_bound = damping/(density * dx_bound * np.pi * self.radius**2)

        boundary_position = np.zeros((3, boundary_n_elem+1)) ## number of nodes = number of elements + 1

        boundary_pos_connect_idx = []

        for i in range(self.num_boundary_threads):
            ## coordinates of current vertex of the polygon i*(rad_n_elem)
            ang_1 = i * angle_between_threads
            pos_1 = self.web_radius * np.array([np.cos(ang_1), np.sin(ang_1), 0.0])

            ## coordinates of next vertex of the polygon 
            ang_2 = (i+1) * angle_between_threads
            pos_2 = self.web_radius * np.array([np.cos(ang_2), np.sin(ang_2), 0.0])

            ## obtaining coordinates of the nodes in between
            dist_array = np.linspace(pos_1, pos_2, rad_n_elem+1)

            boundary_position[..., i*rad_n_elem:(i+1)*rad_n_elem+1] = dist_array.T

            boundary_pos_connect_idx.append(i*rad_n_elem)

        self.boundary_thread = CosseratRod.straight_rod(
            boundary_n_elem,
            boundary_start,
            y_direction,
            normal,
            self.boundary_length,
            base_radius=radius_along_boundary,
            density=density,
            # nu=damping,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
            position = boundary_position
        )

        self.simulator.append(
            self.boundary_thread
        )  # Now rod is ready for simulation, append rod to simulation

        self.simulator.dampen(self.boundary_thread).using(
            AnalyticalLinearDamper, 
            damping_constant = damping_constant_bound,
            time_step = self.sim_dt
        )

        # print(boundary_pos_connect_idx)

        # boundary_pos_connect_idx = tuple(boundary_pos_connect_idx)
        # self.simulator.constrain(self.boundary_thread).using(
        #     FixedConstraint, constrained_position_idx=boundary_pos_connect_idx, constrained_director_idx=boundary_pos_connect_idx
        # )

        for i in range(self.num_rad_threads):
            if i < int(self.num_rad_threads/2):
                self.simulator.connect(first_rod=self.radial_thread[i], second_rod=self.boundary_thread, 
                                       first_connect_idx=-1, second_connect_idx=boundary_pos_connect_idx[i]).using(FixedJoint,
                                                                                                                      k=1e5,
                                                                                                                      nu=100,
                                                                                                                      kt=2e3,
                                                                                                                      nut=0.0)
            else:
                self.simulator.connect(first_rod=self.radial_thread[i-int(self.num_rad_threads/2)], second_rod=self.boundary_thread, 
                                       first_connect_idx=0, second_connect_idx=boundary_pos_connect_idx[i]).using(FixedJoint,
                                                                                                                      k=1e5,
                                                                                                                      nu=100,
                                                                                                                      kt=2e3,
                                                                                                                      nut=0.0)

        theta_final = 2*np.pi*self.spiral_n_turns
        spiral_length = (self.spiral_param_a/2) * (theta_final*np.sqrt(1+theta_final**2) + np.log(theta_final + np.sqrt(1+theta_final**2)))

        spiral_start = self.spiral_offset*x_direction
        radius_along_spiral = np.linspace(radius_tip, radius_tip, self.spiral_n_elem)
        dx_spiral = spiral_length/self.spiral_n_elem
        damping_constant_spiral = damping/(density * dx_spiral * np.pi * self.radius**2)

        spiral_position = np.zeros((3, self.spiral_n_elem+1))

        spiral_pos_connnect_idx = []
        rad_thread_connect_idx = []

        for i in range(self.spiral_n_turns):
            if i < int(self.spiral_n_turns/2):
                n_elem = self.spiral_n_elem_first_half
                adjust_idx = n_elem*self.num_rad_threads*i
            else:
                n_elem = self.spiral_n_elem_second_half
                adjust_idx = self.spiral_n_elem_first_half*self.num_rad_threads*int(self.spiral_n_turns/2) + n_elem*self.num_rad_threads*(i-int(self.spiral_n_turns/2))

            for j in range(self.num_rad_threads):
                ## current coordinates of the intersection of radial thread and spiral
                ang_1 = j*angle_between_threads + i*2*np.pi
                r_1 = self.spiral_param_a*ang_1 + self.spiral_offset
                vec_1 = r_1 * np.array([np.cos(ang_1), np.sin(ang_1), 0.0])

                ## next coordinates of the intersection of radial thread and spiral
                ang_2 = (j+1)*angle_between_threads + i*2*np.pi
                r_2 = self.spiral_param_a*ang_2 + self.spiral_offset
                vec_2 = r_2 * np.array([np.cos(ang_2), np.sin(ang_2), 0.0])

                dist_array = np.linspace(vec_1, vec_2, n_elem+1)

                start_idx = j*n_elem + adjust_idx
                end_idx = start_idx + n_elem + 1

                spiral_position[..., start_idx:end_idx] = dist_array.T

                spiral_pos_connnect_idx.append(start_idx)

                rad_thread_connect_idx.append(int((self.dia_n_elem/(2*self.web_radius))*r_1 + self.dia_n_elem/2))

        # import matplotlib.pyplot as plt
        # plt.plot(spiral_position[0, ...], spiral_position[1, ...], '-o')
        # plt.show()
            
        self.spiral_thread = CosseratRod.straight_rod(
            self.spiral_n_elem,
            spiral_start,
            y_direction,
            normal,
            spiral_length,
            base_radius=radius_along_spiral,
            density=density,
            # nu=damping,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
            position = spiral_position
        )

        self.simulator.append(
            self.spiral_thread
        )  # Now rod is ready for simulation, append rod to simulation

        self.simulator.dampen(self.spiral_thread).using(
            AnalyticalLinearDamper, 
            damping_constant = damping_constant_spiral,
            time_step = self.sim_dt
        )

        # spiral_pos_connnect_idx = tuple(spiral_pos_connnect_idx)
        # self.simulator.constrain(self.spiral_thread).using(
        #     FixedConstraint, constrained_position_idx=spiral_pos_connnect_idx, constrained_director_idx=spiral_pos_connnect_idx
        # )

        j = 0
        for i in range(len(spiral_pos_connnect_idx)):

            if j > 2:
                j = 0

            self.simulator.connect(first_rod=self.radial_thread[j], second_rod=self.spiral_thread, 
                                    first_connect_idx=rad_thread_connect_idx[i], second_connect_idx=spiral_pos_connnect_idx[i]).using(FixedJoint,
                                                                                                                    k=1e5,
                                                                                                                    nu=100,
                                                                                                                    kt=2e3,
                                                                                                                    nut=0.0)
            j = j + 1

        # Call back function to collect arm data from simulation
        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            class WebCallBack(CallBackBaseClass):
                """
                Call back function for Elastica rod
                """

                def __init__(
                    self,
                    step_skip: int,
                    callback_params: dict,
                ):
                    CallBackBaseClass.__init__(self)
                    self.every = step_skip
                    self.callback_params = callback_params

                def make_callback(self, system, time, current_step: int):
                        if current_step % self.every == 0:
                            self.callback_params["time"].append(time)
                            self.callback_params["step"].append(current_step)
                            self.callback_params["position"].append(
                                system.position_collection.copy()
                            )
                            self.callback_params["radius"].append(system.radius.copy())
                            self.callback_params["com"].append(
                                system.compute_position_center_of_mass()
                            )
                            self.callback_params["directors"].append(
                                system.director_collection.copy()
                            )
                            self.callback_params["kappa"].append(system.kappa.copy())
                            self.callback_params["omega_collection"].append(
                                system.omega_collection.copy()
                            )
                            self.callback_params["sigma"].append(system.sigma.copy())
                            self.callback_params["tangents"].append(system.tangents.copy())
                            self.callback_params["velocity_collection"].append(
                                system.velocity_collection.copy()
                            )
                            self.callback_params["acceleration_collection"].append(
                                system.acceleration_collection.copy()
                            )
                            return

            # Collect data using callback function for postprocessing
            self.post_processing_dict_radial_thread = defaultdict(list)
            for i in range(int(self.num_rad_threads/2)):
                self.post_processing_dict_radial_thread_each = defaultdict(list)
                self.simulator.collect_diagnostics(self.radial_thread[i]).using(
                    WebCallBack,
                    step_skip=self.step_skip,
                    callback_params=self.post_processing_dict_radial_thread_each,
                )

                self.post_processing_dict_radial_thread[f"thread {i+1}"].append(self.post_processing_dict_radial_thread_each)

        """Adding forces to the web"""
        for i in range(int(self.num_rad_threads/2)):
            self.simulator.add_forcing_to(self.radial_thread[i]).using(NoForces)
        self.simulator.add_forcing_to(self.boundary_thread).using(NoForces)
        self.simulator.add_forcing_to(self.spiral_thread).using(NoForces)

        """
        heuristic scaling of torque - not this works for a 25:1 slenderness ratio. Bending stiffness scales ~r^4 so if thicker or thinner, 
        the torque value might need to be adjust_idxed. This scaling is more designed for changing the youngs modulus 
        """
        self.alpha = self.torque_scale * self.radius * self.youngs_modulus
        # Add muscle torques acting on the arm for actuation
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected by RL to
        # generate torques along the arm.
        # self.torque_profile_list_for_muscle_in_normal_14_dir = defaultdict(list)
        self.spline_points_func_array_normal_14_dir = []
        # self.simulator.add_forcing_to(self.shearable_rod).using(
        #     MuscleTorquesWithVaryingBetaSplines,
        #     base_length=self.base_length,
        #     number_of_control_points=self.number_of_control_points,
        #     points_func_array=self.spline_points_func_array_normal_14_dir,
        #     muscle_torque_scale=self.alpha,
        #     direction_14=str("normal_14"),
        #     step_skip=self.step_skip,
        #     max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
        #     # max_signal_rate_of_change=4*self.sim_dt,
        #     torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_14_dir,
        # )

        # self.torque_profile_list_for_muscle_in_binormal_14_dir = defaultdict(list)
        self.spline_points_func_array_binormal_14_dir = []
        # self.simulator.add_forcing_to(self.shearable_rod).using(
        #     MuscleTorquesWithVaryingBetaSplines,
        #     base_length=self.base_length,
        #     number_of_control_points=self.number_of_control_points,
        #     points_func_array=self.spline_points_func_array_binormal_14_dir,
        #     muscle_torque_scale=self.alpha,
        #     direction_14=str("binormal_14"),
        #     step_skip=self.step_skip,
        #     max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
        #     # max_signal_rate_of_change=4*self.sim_dt,
        #     torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_14_dir,
        # )

        ###--------------GENERATE TARGET TRAJECTORY--------------###
        self.tick = 0


        if self.mode == 2:
            # TODO: change this to a object that will update thee target position as you go
            target_trajectory = generate_trajectory(
                self.max_episode_final_time, self.sim_dt, self.target_v_scale
            )
            # print(target_trajectory)
            # import matplotlib.pyplot as plt
            # plt.plot(target_trajectory)
            # plt.show()
            
            self.wsol = target_trajectory
        elif self.mode == 1:
            end_time = self.max_episode_final_time * 1.1
            numpoints = np.rint(1 / self.sim_dt * end_time).astype(int)
            self.wsol = np.zeros((numpoints, 3))
            self.wsol[:] = self.target_location

        velocity_sphere = self.wsol * 0
        velocity_sphere[:-1] = (self.wsol[1:] - self.wsol[:-1]) / self.sim_dt
        velocity_sphere[-1] = velocity_sphere[-2]
        self.velocity_sphere = velocity_sphere

        ###--------------ADD SPIDER TO SIMULATION--------------###
        # self.sphere_radius = 0.025 * 1000
        # target_position = self.wsol[0]
        # # print(target_position)
        # # initialize sphere
        # self.sphere = Sphere(
        #     center=target_position,  # initialize target position of the ball
        #     base_radius=self.sphere_radius,
        #     density=1000 * 1e-6,
        # )
        # self.simulator.append(self.sphere)
        # self.sphere.velocity_collection[..., 0] = self.velocity_sphere[0]

        # if self.COLLECT_DATA_FOR_POSTPROCESSING:

        #     # Call back function to collect target sphere data from simulation
        #     class RigidSphereCallBack(CallBackBaseClass):
        #         """
        #         Call back function for target sphere
        #         """

        #         def __init__(self, step_skip: int, callback_params: dict):
        #             CallBackBaseClass.__init__(self)
        #             self.every = step_skip
        #             self.callback_params = callback_params

        #         def make_callback(self, system, time, current_step: int):
        #             if current_step % self.every == 0:
        #                 self.callback_params["time"].append(time)
        #                 self.callback_params["step"].append(current_step)
        #                 self.callback_params["position"].append(
        #                     system.position_collection.copy()
        #                 )
        #                 self.callback_params["velocity"].append(
        #                     system.velocity_collection.copy()
        #                 )
        #                 self.callback_params["radius"].append(
        #                     copy.deepcopy(system.radius)
        #                 )
        #                 self.callback_params["com"].append(
        #                     system.compute_position_center_of_mass()
        #                 )
        #                 return

        #     self.post_processing_dict_sphere = defaultdict(list)
        #     self.simulator.collect_diagnostics(self.sphere).using(
        #         RigidSphereCallBack,
        #         step_skip=self.step_skip,
        #         callback_params=self.post_processing_dict_sphere,
        #     )

        ###--------------FINALIZE SIMULATION--------------###
        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )
        self.time_tracker = np.float32(0.0)

        state = self.get_state()
        self._target = self.wsol[self.tick]

        if return_info:
            return state, {}
        else:
            return state
        # return np.array(state, dtype=np.float32)

    def save_data(self, save_data, filename_data, save_video, filename_video):

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            # save_file = pd.DataFrame({"Rod Tip Position":[], "Rod Element Position":[], "Rod Radius":[],
            #                             "Rod Directors": [], "Rod Kappa": [], "Rod Omega": [], "Rod Sigma": [],
            #                             "Rod Tangents": [], "Rod Velocity": [], "Rod Acceleration": [],
            #                             "Rod elements": [], "Sphere Position": [], "Sphere Velocity": [], 
            #                             "Sphere Radius": []})

            position_rod = np.array(self.post_processing_dict_thread_1["position"])
            # print(position_rod[0, ...])
            position_sphere = np.array(self.post_processing_dict_sphere["position"])
            # print(self.post_processing_dict_thread_1["time"])
            
            position_rod_tip = []
            for i, dict_data in enumerate(position_rod):
                # print(i)
                position_rod_tip.append(dict_data[...,-1])

                # temp_df = pd.DataFrame({"Rod Tip Position":[position_rod_tip[i]], "Rod Element Position":[position_rod[i]], "Rod Radius":[radii_rod[i]],
                #                         "Rod Directors": [directors_rod[i]], "Rod Kappa": [kappa_rod[i]], "Rod Omega": omega_rod[i], "Rod Sigma": [sigma_rod[i]],
                #                         "Rod Tangents": [tangents_rod[i]], "Rod Velocity": [velocity_rod[i]], "Rod Acceleration": [acceleration_rod[i]],
                #                         "Rod elements": [n_elems_rod], "Sphere Position": [position_sphere[i]], "Sphere Velocity": [velocity_sphere[i]], 
                #                         "Sphere Radius": [radii_sphere[i]]})
                # save_file = pd.concat([save_file, temp_df], ignore_index=True)

            if save_data:

                np.savez(
                    file=filename_data,
                    position_rod_tip=position_rod_tip,
                    position_rod=position_rod,
                    radii_rod = np.array(self.post_processing_dict_thread_1["radius"]),
                    directors_rod=np.array(self.post_processing_dict_thread_1["directors"]),
                    kappa_rod=np.array(self.post_processing_dict_thread_1["kappa"]),
                    omega_rod=np.array(self.post_processing_dict_thread_1["omega_collection"]),
                    sigma_rod=np.array(self.post_processing_dict_thread_1["sigma"]),
                    tangents_rod=np.array(self.post_processing_dict_thread_1["tangents"]),
                    velocity_rod=np.array(self.post_processing_dict_thread_1["velocity_collection"]),
                    acceleration_rod=np.array(self.post_processing_dict_thread_1["acceleration_collection"]),
                    n_elems_rod=self.shearable_rod.n_elems,

                    position_sphere=position_sphere,
                    velocity_sphere=np.array(self.post_processing_dict_sphere["velocity"]),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"])
                    ) 

            # save_file.to_csv("trial_save_data.csv", index=False)

            minx = min(np.min(position_rod[2]), np.min(position_sphere[2])) - 10
            maxx = max(np.max(position_rod[2]), np.max(position_sphere[2])) + 10

            miny = min(np.min(position_rod[0]), np.min(position_sphere[0])) - 10
            maxy = max(np.max(position_rod[0]), np.max(position_sphere[0])) + 10

            minz = min(np.min(position_rod[1]), np.min(position_sphere[1])) - 10
            maxz = max(np.max(position_rod[1]), np.max(position_sphere[1])) + 10

            limits = [minx, maxx, miny, maxy, minz, maxz]
            # print(limits)

            if save_video:
                # save_video_obj = save_video(maxwidth, aspect_ratio, self._target.tolist(), self.sphere_radius)
                plot_video_with_sphere(
                    [self.post_processing_dict_thread_1],
                    # [self.post_processing_dict_sphere],
                    video_name=filename_video,
                    fps=self.rendering_fps,
                    step=1, limits=limits)

    def render(self, mode="human", close=False):
        maxwidth = 800
        aspect_ratio = 3 / 4

        if self.viewer is None:
            from gym_fiber.utils.render import pyglet_rendering
            self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)

        if self.renderer is None:
            # Switch renderer depending on configuration
            if RENDERER_CONFIG == RendererType.POVRAY:
                from gym_fiber.utils.render.povray_renderer import Session
            elif RENDERER_CONFIG == RendererType.MATPLOTLIB:
                from gym_fiber.utils.render.matplotlib_renderer import Session
            else:
                raise NotImplementedError("Rendering module is not imported properly")
            assert issubclass(
                Session, BaseRenderer
            ), "Rendering module is not properly subclassed"
            assert issubclass(
                Session, BaseElasticaRendererSession
            ), "Rendering module is not properly subclassed"
            self.renderer = Session(width=maxwidth, height=int(maxwidth*aspect_ratio))
            if RENDERER_CONFIG == RendererType.POVRAY:
                self.renderer.add_rod(self.radial_thread)
                self.renderer.add_rod(self.radial_thread)
            elif RENDERER_CONFIG == RendererType.MATPLOTLIB:
                for i in range(int(self.num_rad_threads/2)):
                    self.renderer.add_rod(self.radial_thread[i], 'rad')
                # for i in range(self.num_boundary_threads):
                self.renderer.add_rod(self.boundary_thread, 'bound')
                self.renderer.add_rod(self.spiral_thread, 'spiral')
            else:
                raise NotImplementedError("Rendering module is not imported properly")
              
            # self.renderer.add_point(self._target.tolist(), self.sphere_radius)

        # POVRAY
        if RENDERER_CONFIG == RendererType.POVRAY:
            state_image = self.renderer.render(
                maxwidth, int(maxwidth * aspect_ratio * 0.7),
                camera_param=("location", [2000, 0, 2000], "look_at", [0, 0, 0])
            )
        elif RENDERER_CONFIG == RendererType.MATPLOTLIB:
            state_image = self.renderer.render()
        else:
            raise NotImplementedError("Rendering module is not imported properly")

        self.viewer.imshow(state_image)

        return state_image

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
