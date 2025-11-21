from collections import defaultdict
import time
import copy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from typing import Optional, Sequence

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *
from elastica.external_forces import GravityForces
from elastica.modules.damping import Damping

#fiber_v1.fiber.
from gym_fiber.utils.custom_elastica.pointforce import PointForce, PointForceSinsusoidal
from gym_fiber.utils.render import pyglet_rendering
from gym_fiber.utils.render.matplotlib_renderer import Session
from gym_fiber.utils.render.post_processing import plot_network_video, plot_network_video_2D

from tqdm import tqdm

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass

def render_web(horizontal_thread, num_horizontal_threads, vertical_thread, num_vertical_threads, viewer, renderer):

    for i in range(num_horizontal_threads):
        renderer.add_rod(horizontal_thread[i], 'rad')

    for i in range(num_vertical_threads):
        renderer.add_rod(vertical_thread[i], 'rad')

    state_image = renderer.render()

    viewer.imshow(state_image)

    return state_image

def run_and_update_plot(simulator, StatefulStepper, sim_dt, start_time, stop_time, viewer, renderer, n_steps):

    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, simulator)

    # n_steps = int((stop_time - start_time) / sim_dt)
    time = start_time
    
    for i in tqdm(range(n_steps)):
        time = do_step(StatefulStepper, stages_and_updates, simulator, time, sim_dt)

        for j in range(num_horizontal_threads):
            horizontal_bending_energy[j, i] = CosseratRod.compute_bending_energy(horizontal_thread[j])
            horizontal_shear_energy[j, i] = CosseratRod.compute_shear_energy(horizontal_thread[j])
            horizontal_translational_energy[j, i] = CosseratRod.compute_translational_energy(horizontal_thread[j])
            horizontal_rotational_energy[j, i] = CosseratRod.compute_rotational_energy(horizontal_thread[j])
        for k in range(num_vertical_threads):
            vertical_bending_energy[k, i] = CosseratRod.compute_bending_energy(vertical_thread[k])
            vertical_shear_energy[k, i] = CosseratRod.compute_shear_energy(vertical_thread[k])
            vertical_translational_energy[k, i] = CosseratRod.compute_translational_energy(vertical_thread[k])
            vertical_rotational_energy[k, i] = CosseratRod.compute_rotational_energy(vertical_thread[k])
    # render_web(horizontal_thread, num_horizontal_threads, vertical_thread, num_vertical_threads, viewer, renderer)
    return time

'''Fiber network geometry'''
num_horizontal_threads = 4
num_vertical_threads = 4
network_origin = np.zeros((3,))

"""COMMON THREAD PROPERTIES"""
thread_length = 1000e-3 * 1e3
thread_radius = 0.5 * 4e-3 * 1e3
dx = 10
n_elem = np.rint(thread_length/dx).astype(int) #int(thread_length/dx)

'''Young's modulus in MPa (scaled SI)'''
youngs_modulus = 228e6
poisson_ratio = 0.45

'''Shear modulus in MPa (scaled SI)'''
shear_modulus = 0.5 * youngs_modulus / (poisson_ratio + 1.0) #0.98 * 1e6 #
# poisson_ratio = 1 - 0.5 * youngs_modulus/shear_modulus

'''Density in g/mm3 (scaled SI)'''
density = 1000 * 1e-6

'''Damping Constant in 1/s'''
damping_constant = 20.0

'''Physical simulation timestep in seconds'''
sim_dt = 5e-6

'''Simulation time in seconds (scaled SI)'''
max_episode_final_time = 5                    

'''number of timesteps to be taken before updating the RL algorithm'''
update_interval = 0.1 
num_steps_per_update = np.rint(
    update_interval / sim_dt
).astype(int)

tension_force = 2e5

COLLECT_DATA_FOR_POSTPROCESSING = True
SAVE_VIDEO = True

rendering_fps = 1000 # 200
step_skip = np.rint(1.0 / (rendering_fps * sim_dt)).astype(int)

StatefulStepper = PositionVerlet()

simulator = BaseSimulator()

normal = np.array([0.0, 0.0, 1.0])
x_direction = np.array([1.0, 0.0, 0.0])
y_direction = np.array([0.0, 1.0, 0.0])

"""HORIZONTAL THREADS"""
horizontal_thread = [None for i in range(num_horizontal_threads)]

if num_horizontal_threads == 1:
    horizontal_thread_y_pos = np.linspace(network_origin[1], network_origin[1], num_horizontal_threads+2)
else:
    horizontal_thread_y_pos = np.linspace(network_origin[1]-thread_length/2, network_origin[1]+thread_length/2, num_horizontal_threads+2)

horizontal_thread_y_pos = horizontal_thread_y_pos[1:-1]
# horizontal_thread_y_pos[0] = 100 #-100
# horizontal_thread_y_pos[-1] = 400 #200
vert_connect_idx = np.linspace(0, n_elem+1, num_horizontal_threads+2)
vert_connect_idx = vert_connect_idx[1:-1]
# vert_connect_idx[0] = int(n_elem * 0.4)
# vert_connect_idx[-1] = int(n_elem * 0.7)
# print(vert_connect_idx)

horizontal_tension_force = np.array([tension_force, 0.0, 0.0])

for i in range(num_horizontal_threads):

    horizontal_thread_start = network_origin - 0.5 * thread_length * x_direction
    horizontal_thread_start[1] = horizontal_thread_y_pos[i] 
    # print(horizontal_thread_start)   

    horizontal_thread[i] = CosseratRod.straight_rod(
        n_elements=n_elem,
        start=horizontal_thread_start,
        direction=x_direction,
        normal=y_direction,
        base_length=thread_length,
        base_radius=thread_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        # shear_modulus=shear_modulus
    )

    simulator.append(
        horizontal_thread[i]
    )

    simulator.dampen(horizontal_thread[i]).using(
        AnalyticalLinearDamper, 
        damping_constant = damping_constant,
        time_step = sim_dt
    )

    simulator.dampen(horizontal_thread[i]).using(
        LaplaceDissipationFilter,
        filter_order = 6
    )

    simulator.constrain(horizontal_thread[i]).using(
        GeneralConstraint, constrained_position_idx=(0, -1), constrained_director_idx=(0, -1),
        translational_constraint_selector=np.array([False, True, True]), 
        rotational_constraint_selector=np.array([True, True, True])
    )

    # simulator.constrain(horizontal_thread[i]).using(
    #     GeneralConstraint, constrained_position_idx=(-1, ), constrained_director_idx=(),
    #     translational_constraint_selector=np.array([True, True, True]), 
    #     rotational_constraint_selector=np.array([True, True, True])
    # )

    simulator.add_forcing_to(horizontal_thread[i]).using(
        EndpointForces, -horizontal_tension_force, horizontal_tension_force, ramp_up_time=0.25
    )

#int(n_elem/2.0)-5

"""VERTICAL THREADS"""
vertical_thread = [None for i in range(num_vertical_threads)]

if num_vertical_threads == 1:
    vertical_thread_x_pos = np.linspace(network_origin[0], network_origin[0], num_vertical_threads+2)
else:
    vertical_thread_x_pos = np.linspace(network_origin[0]-thread_length/2, network_origin[0]+thread_length/2, num_horizontal_threads+2)

vertical_thread_x_pos = vertical_thread_x_pos[1:-1]
# vertical_thread_x_pos[0] = 300 #-200
# vertical_thread_x_pos[-1] = 800 #300
hor_connect_idx = np.linspace(0, n_elem+1, num_vertical_threads+2)
hor_connect_idx = hor_connect_idx[1:-1]
# hor_connect_idx[0] = int(n_elem * 0.3)
# hor_connect_idx[-1] = int(n_elem * 0.8)
# print(hor_connect_idx)

vertical_tension_force = np.array([0.0, tension_force, 0.0])

for i in range(num_vertical_threads):

    vertical_thread_start = network_origin - 0.5 * thread_length * y_direction
    vertical_thread_start[0] = vertical_thread_x_pos[i]
    # print(vertical_thread_start)
    
    vertical_thread[i] = CosseratRod.straight_rod(
        n_elements=n_elem,
        start=vertical_thread_start,
        direction=y_direction,
        normal=x_direction,
        base_length=thread_length,
        base_radius=thread_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        # shear_modulus=shear_modulus
    )

    simulator.append(
        vertical_thread[i]
    )  

    simulator.dampen(vertical_thread[i]).using(
        AnalyticalLinearDamper, 
        damping_constant = damping_constant,
        time_step = sim_dt
    )

    simulator.dampen(vertical_thread[i]).using(
        LaplaceDissipationFilter,
        filter_order = 6
    )

    simulator.constrain(vertical_thread[i]).using(
        GeneralConstraint, constrained_position_idx=(0, ), constrained_director_idx=(0, -1),
        translational_constraint_selector=np.array([True, False, True]), 
        rotational_constraint_selector=np.array([True, True, True])
    )

    # if i == 0:
    simulator.constrain(vertical_thread[i]).using(
        GeneralConstraint, constrained_position_idx=(-1, ), constrained_director_idx=(),
        translational_constraint_selector=np.array([True, True, True]), 
        rotational_constraint_selector=np.array([True, True, True])
    )
    # else:
    #     simulator.constrain(vertical_thread[i]).using(
    #         GeneralConstraint, constrained_position_idx=(-1, ), constrained_director_idx=(0, -1),
    #         translational_constraint_selector=np.array([True, True, True]), 
    #         rotational_constraint_selector=np.array([True, True, True])
    #     )

    simulator.add_forcing_to(vertical_thread[i]).using(
        EndpointForces, -vertical_tension_force, vertical_tension_force, ramp_up_time=0.25
    )

# spheroid_center = horizontal_thread[0].position_collection[...,int(hor_connect_idx[0])]
# spheroid_radius = 2 * thread_radius * 5
# spheroid_density = density

# spheroid = Sphere(
#      spheroid_center, 
#      spheroid_radius, 
#      spheroid_density,
# )
# simulator.append(spheroid)

# simulator.connect(
#     horizontal_thread[0], spheroid,
#     int(hor_connect_idx[0]), 0
#     ).using(FreeJoint,
#             k = 1e9,
#             nu = 0.0)

for j in range(num_horizontal_threads):
    for i in range(num_vertical_threads):
        rest_rotation_matrix = horizontal_thread[j].director_collection[..., int(hor_connect_idx[i])] @ vertical_thread[i].director_collection[..., int(vert_connect_idx[j])]
        simulator.connect(
            horizontal_thread[j], vertical_thread[i],
            int(hor_connect_idx[i]), int(vert_connect_idx[j])
            ).using(FixedJoint,
                    k=1e9, #0.5*1e6,
                    nu=0.0,
                    kt=1e9,
                    rest_rotation_matrix=rest_rotation_matrix
                    )
        
point_force = np.array([0.0, -0.2e6, 0.0])
point_force_spread = np.arange(-2,2+1)
stencil = 1/(np.abs(point_force_spread)+1)
stencil /= np.sum(stencil)
node_idx = np.rint(n_elem/2.).astype(int)
for i in point_force_spread:
    simulator.add_forcing_to(horizontal_thread[-1]).using(
        PointForceSinsusoidal, node_idx+i,  point_force*stencil[i], ramp_up_time=1.0, hold_time=0.5,
    )

if COLLECT_DATA_FOR_POSTPROCESSING:
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
                    self.callback_params["internal_forces"].append(
                        system.internal_forces.copy()
                    )
                    self.callback_params["internal_torques"].append(
                        system.internal_torques.copy()
                    )
                    self.callback_params["angular_acceleration"].append(
                        system.alpha_collection.copy()
                    )
                    self.callback_params["external_forces"].append(
                        system.external_forces.copy()
                    )
                    self.callback_params["external_torques"].append(
                        system.external_torques.copy()
                    )
                    return
                
    # Collect data using callback function for postprocessing
    post_processing_dict_horizontal_thread = []
    for i in range(num_horizontal_threads):
        post_processing_dict_horizontal_thread_each = defaultdict(list)
        simulator.collect_diagnostics(horizontal_thread[i]).using(
            WebCallBack,
            step_skip=step_skip,
            callback_params=post_processing_dict_horizontal_thread_each,
        )

        post_processing_dict_horizontal_thread.append(post_processing_dict_horizontal_thread_each)

    post_processing_dict_vertical_thread = []
    for i in range(num_vertical_threads):
        post_processing_dict_vertical_thread_each = defaultdict(list)
        simulator.collect_diagnostics(vertical_thread[i]).using(
            WebCallBack,
            step_skip=step_skip,
            callback_params=post_processing_dict_vertical_thread_each,
        )

        post_processing_dict_vertical_thread.append(post_processing_dict_vertical_thread_each)

simulator.finalize()

current_time = 0.0
n_steps = np.rint((max_episode_final_time ) / sim_dt).astype(int)

horizontal_bending_energy = np.zeros((num_horizontal_threads, n_steps))
horizontal_shear_energy = np.zeros((num_horizontal_threads, n_steps))
horizontal_translational_energy = np.zeros((num_horizontal_threads, n_steps))
horizontal_rotational_energy = np.zeros((num_horizontal_threads, n_steps))

vertical_bending_energy = np.zeros((num_vertical_threads, n_steps))
vertical_shear_energy = np.zeros((num_vertical_threads, n_steps))
vertical_translational_energy = np.zeros((num_vertical_threads, n_steps))
vertical_rotational_energy = np.zeros((num_vertical_threads, n_steps))

do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, simulator)

for i in tqdm(range(n_steps)):
    current_time = do_step(StatefulStepper, stages_and_updates, simulator, current_time, sim_dt)

    for j in range(num_horizontal_threads):
        horizontal_bending_energy[j, i] = CosseratRod.compute_bending_energy(horizontal_thread[j])
        horizontal_shear_energy[j, i] = CosseratRod.compute_shear_energy(horizontal_thread[j])
        horizontal_translational_energy[j, i] = CosseratRod.compute_translational_energy(horizontal_thread[j])
        horizontal_rotational_energy[j, i] = CosseratRod.compute_rotational_energy(horizontal_thread[j])
    for k in range(num_vertical_threads):
        vertical_bending_energy[k, i] = CosseratRod.compute_bending_energy(vertical_thread[k])
        vertical_shear_energy[k, i] = CosseratRod.compute_shear_energy(vertical_thread[k])
        vertical_translational_energy[k, i] = CosseratRod.compute_translational_energy(vertical_thread[k])
        vertical_rotational_energy[k, i] = CosseratRod.compute_rotational_energy(vertical_thread[k])

rods_history = post_processing_dict_horizontal_thread + post_processing_dict_vertical_thread
video_name = f"network_{num_horizontal_threads+num_vertical_threads}rods_4by4.mp4"
x_limits = [-thread_length*0.1, thread_length*1.1]
y_limits = [-thread_length*0.4, thread_length*0.8]
x_limits = (-thread_length/2-5, thread_length/2+5)
y_limits = (-thread_length/2-5, thread_length/2+5)
plot_network_video_2D(rods_history=rods_history, video_name=video_name, fps=rendering_fps, step=1, x_limits=x_limits, y_limits=y_limits)

np.savez(
    file = "fiber_network_4by4.npz",
    rods_history = np.array([rods_history])
)