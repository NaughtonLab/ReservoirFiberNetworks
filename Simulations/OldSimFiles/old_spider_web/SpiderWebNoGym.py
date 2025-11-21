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

#spider_web_v1.spider_web.
from gym_spider_web.utils.render import pyglet_rendering
from gym_spider_web.utils.render.matplotlib_renderer import Session
from gym_spider_web.utils.render.post_processing import plot_web_video

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass

def render_web(radial_thread, boundary_thread, spiral_thread, num_rad_threads, viewer, renderer):

    for i in range(int(num_rad_threads/2)):
        renderer.add_rod(radial_thread[i], 'rad')
    # for i in range(num_boundary_threads):
    renderer.add_rod(boundary_thread, 'bound')
    # renderer.add_rod(spiral_thread, 'spiral')

    state_image = renderer.render()

    viewer.imshow(state_image)

    return state_image

def run_and_update_plot(simulator, StatefulStepper, sim_dt, start_time, stop_time, viewer, renderer):

    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, simulator)

    n_steps = int((stop_time - start_time) / sim_dt)
    time = start_time
    for i in range(n_steps):
        time = do_step(StatefulStepper, stages_and_updates, simulator, time, sim_dt)
    # render_web(radial_thread, boundary_thread, spiral_thread, num_rad_threads, viewer, renderer)
    return time

'''Web geometry'''
# The number of radial threads should be the number of lines starting from origin and extending till the web boundary
# Thus for a hexagonal web the number of radial threads will be 6
# Stick with even number of radial threads for now
num_rad_threads = 6
num_boundary_threads = num_rad_threads
web_origin = np.zeros((3,))
web_radius = 1000/5
web_plane = 'xy'
boundary_length = (web_radius)*num_boundary_threads
# If web_radius reduces by a factor of n then reduce the spiral_offset and spiral_param_a by the same factor n to keep the geometry constant
spiral_offset = 100/5
spiral_param_a = 22/5
spiral_n_turns = 6

'''number of elements in each type of thread'''
dia_n_elem = 150 #200 # This number indicates the number of elements in the diameter of the web. Half of these + 1 will be the number of elements in each side of the polygon.
spiral_n_elem_first_half = 50
spiral_n_elem_second_half = 100
spiral_n_elem = (spiral_n_elem_first_half + spiral_n_elem_second_half) * int(spiral_n_turns/2) * num_rad_threads

'''Physical simulation timestep in seconds'''
sim_dt = 1e-5 #2*1e-6 #1e-5 #or 1e-4

'''number of timesteps to be taken before updating the RL algorithm'''
update_interval = 0.1 
num_steps_per_update = np.rint(
    update_interval / sim_dt
).astype(int)

'''Young's modulus in MPa (scaled SI)'''
youngs_modulus = 1e6
poisson_ratio = 0.45

'''Shear modulus in MPa (scaled SI)'''
shear_modulus = 0.5 * youngs_modulus / (poisson_ratio + 1.0) #0.98 * 1e6 #
# poisson_ratio = 1 - 0.5 * youngs_modulus/shear_modulus

'''Simulation time in seconds (scaled SI)'''
max_episode_final_time = 10

'''Defining parameters of the each silk thread in mm (scaled SI)'''             
radial_thread_radius = 25.0   
boundary_thread_radius = 5.0 / 100                

COLLECT_DATA_FOR_POSTPROCESSING = True
SAVE_VIDEO = True

rendering_fps = 200
step_skip = np.rint(1.0 / (rendering_fps * sim_dt)).astype(int)

add_ext_forcing = False

StatefulStepper = PositionVerlet()

simulator = BaseSimulator()

if web_plane == 'xy':
    normal = np.array([0.0, 0.0, 1.0])
else:
    print("Plane of the Spider Web is not XY. Add a new configuration or define the plane")
x_direction = np.array([1.0, 0.0, 0.0])
y_direction = np.array([0.0, 1.0, 0.0])

angle_between_threads = 2 * np.pi / num_rad_threads

"""RADIAL THREADS"""
# Set the thread properties after defining rods

damping = youngs_modulus * 1e-3 #10 #youngs_modulus * 1e-9 
density = 1300 * 1e-6
radius_along_diameter = np.linspace(radial_thread_radius, radial_thread_radius, dia_n_elem)
dx_dia = 2*web_radius/dia_n_elem
damping_constant_rad = 2*1e4 #damping/(density * dx_dia * np.pi * radius**2)

radial_thread = [None for i in range(num_rad_threads)]

for i in range(int(num_rad_threads/2)):
    ang = i * angle_between_threads
    direction = np.array([np.cos(ang), np.sin(ang), 0.0])
    start = web_origin - web_radius * direction

    zero_array = np.zeros((3, dia_n_elem+1))

    radial_thread[i] = CosseratRod.straight_rod(
        dia_n_elem,
        start,
        direction,
        normal,
        base_length=2*web_radius,
        base_radius=radius_along_diameter,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
        # velocities = zero_array
        # internal_forces = radial_thread_internal_forces
    )

    simulator.append(
        radial_thread[i]
    )  # Now rod is ready for simulation, append rod to simulation

    # # print(f"thread {i} added")

    simulator.dampen(radial_thread[i]).using(
        AnalyticalLinearDamper, 
        damping_constant = damping_constant_rad,
        time_step = sim_dt
    )

    simulator.constrain(radial_thread[i]).using(
        FixedConstraint, constrained_position_idx=(0,-1), constrained_director_idx=(0,-1)
    )

    if i > 0:
        simulator.connect(radial_thread[i-1], radial_thread[i],
                          int(dia_n_elem/2), int(dia_n_elem/2)).using(FixedJoint,
                                                                      k=10.0,
                                                                      nu=0.0,
                                                                      kt=0.0)
        
    # simulator.add_forcing_to(radial_thread[i]).using(GravityForces, acc_gravity=np.array([0.0, 0.0, 9.81*1e3]))

"""
The simulation is not blowing up when we have
1. only radial threads each 2000mm long and 5mm in radius, n_elem = 100 
2. sim_dt = 1e-4, Y=1MPa, poisson's ratio = 0.45, damping constant = 100
3. each thread is constrained at its boundaries
The simulation blows up when sim_dt = 1e-2 or 1e-3

The following parameters are working for sim_dt=1e-2, -3, -4 (with and without gravity) only with damping
radial thread length = 400mm
radial thread radius = 0.05mm
number of elements = 120
Y = 1MPa, poisson's ratio = 0.45
damping constant = 5 x 10^3
contact params: k=10, nu=0, kt=0

Why are these parameters not working without damping and external forces?
- we have internal forces, torques, stresses and couples of inclined rods = nan and that of horizontal rod = 0
- same observation for bending energy, shear energy, translational energy, and rotational energy
- the external forces and torques of all three rods are 0
"""

"""BOUNDARY THREADS"""

rad_n_elem = int(dia_n_elem/2)
n_node_side = rad_n_elem+1
boundary_n_node = (n_node_side-1)*num_boundary_threads + 1
boundary_n_elem = boundary_n_node - 1

radius_along_boundary = np.linspace(boundary_thread_radius, boundary_thread_radius, boundary_n_elem)
dx_bound = boundary_length/boundary_n_elem
damping_constant_bound = 200

boundary_position = np.zeros((3, boundary_n_node)) ## number of nodes = number of elements + 1

boundary_pos_connect_idx = []

for i in range(num_boundary_threads):
    ## coordinates of current vertex of the polygon i*(rad_n_elem)
    ang_1 = i * angle_between_threads
    pos_1 = web_radius * np.array([np.cos(ang_1), np.sin(ang_1), 0.0])

    ## coordinates of next vertex of the polygon 
    ang_2 = (i+1) * angle_between_threads
    pos_2 = web_radius * np.array([np.cos(ang_2), np.sin(ang_2), 0.0])

    ## obtaining coordinates of the nodes in between
    dist_array = np.linspace(pos_1, pos_2, n_node_side)
    dist_array = dist_array.T
    if i < (num_boundary_threads - 1):
        dist_array = dist_array[..., :-1]

    if i == 0:
        boundary_start = pos_1
        boundary_position = dist_array
        boundary_pos_connect_idx.append(i)
        idx = boundary_position.shape[1] - 1
        boundary_pos_connect_idx.append(idx)
    else:
        boundary_position = np.concatenate((boundary_position, dist_array), axis = 1)
        idx = boundary_position.shape[1] - 1
        boundary_pos_connect_idx.append(idx)

    # boundary_position[..., i*(rad_n_elem+1):(i+1)*(rad_n_elem+1)+1] = dist_array.T
    # boundary_position = np.concatenate((dist_array, boundary_position), axis = 1)

    # boundary_pos_connect_idx.append(i*n_node_side)

boundary_thread = CosseratRod.straight_rod(
    boundary_n_elem,
    boundary_start,
    y_direction,
    normal,
    boundary_length,
    base_radius=radius_along_boundary,
    density=density,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
    position = boundary_position
)

simulator.append(
    boundary_thread
)  # Now rod is ready for simulation, append rod to simulation

simulator.dampen(boundary_thread).using(
    AnalyticalLinearDamper, 
    damping_constant = damping_constant_bound,
    time_step = sim_dt
)

# print(boundary_pos_connect_idx)
# boundary_pos_connect_idx = boundary_pos_connect_idx[0:-2]

# boundary_pos_connect_idx = tuple(boundary_pos_connect_idx)
# simulator.constrain(boundary_thread).using(
#     FixedConstraint, constrained_position_idx=boundary_pos_connect_idx, constrained_director_idx=boundary_pos_connect_idx
# )

for i in range(num_rad_threads):
    if i < int(num_rad_threads/2):
        simulator.connect(first_rod=radial_thread[i], second_rod=boundary_thread,
                          first_connect_idx=-1, second_connect_idx=boundary_pos_connect_idx[i]).using(FixedJoint,
                                                                                                      k=1.0,
                                                                                                      nu=0.0,
                                                                                                      kt=0.0)
    else:
        simulator.connect(first_rod=radial_thread[i-int(num_rad_threads/2)], second_rod=boundary_thread,
                          first_connect_idx=0, second_connect_idx=boundary_pos_connect_idx[i]).using(FixedJoint,
                                                                                                     k=1.0,
                                                                                                     nu=0.0,
                                                                                                     kt=0.0)
        
simulator.connect(first_rod=radial_thread[0], second_rod=boundary_thread,
                  first_connect_idx=-1, second_connect_idx=-1).using(FixedJoint,
                                                                    k=1.0,
                                                                    nu=0.0,
                                                                    kt=0.0)
        
"""SPIRAL THREADS"""

theta_final = 2*np.pi*spiral_n_turns
spiral_length = (spiral_param_a/2) * (theta_final*np.sqrt(1+theta_final**2) + np.log(theta_final + np.sqrt(1+theta_final**2)))

spiral_start = spiral_offset*x_direction
radius_along_spiral = np.linspace(radial_thread_radius, radial_thread_radius, spiral_n_elem)
dx_spiral = spiral_length/spiral_n_elem
damping_constant_spiral = damping/(density * dx_spiral * np.pi * radial_thread_radius**2)

spiral_position = np.zeros((3, spiral_n_elem+1))

spiral_pos_connect_idx = []
rad_thread_connect_idx = []

for i in range(spiral_n_turns):
    if i < int(spiral_n_turns/2):
        n_elem = spiral_n_elem_first_half
        adjust_idx = n_elem*num_rad_threads*i
    else:
        n_elem = spiral_n_elem_second_half
        adjust_idx = spiral_n_elem_first_half*num_rad_threads*int(spiral_n_turns/2) + n_elem*num_rad_threads*(i-int(spiral_n_turns/2))

    for j in range(num_rad_threads):
        ## current coordinates of the intersection of radial thread and spiral
        ang_1 = j*angle_between_threads + i*2*np.pi
        r_1 = spiral_param_a*ang_1 + spiral_offset
        vec_1 = r_1 * np.array([np.cos(ang_1), np.sin(ang_1), 0.0])

        ## next coordinates of the intersection of radial thread and spiral
        ang_2 = (j+1)*angle_between_threads + i*2*np.pi
        r_2 = spiral_param_a*ang_2 + spiral_offset
        vec_2 = r_2 * np.array([np.cos(ang_2), np.sin(ang_2), 0.0])

        dist_array = np.linspace(vec_1, vec_2, n_elem+1)

        start_idx = j*n_elem + adjust_idx
        end_idx = start_idx + n_elem + 1

        spiral_position[..., start_idx:end_idx] = dist_array.T

        spiral_pos_connect_idx.append(start_idx)

        rad_thread_connect_idx.append(int((dia_n_elem/(2*web_radius))*r_1 + dia_n_elem/2))

# import matplotlib.pyplot as plt
# plt.plot(spiral_position[0, ...], spiral_position[1, ...], '-o')
# plt.show()
    
spiral_thread = CosseratRod.straight_rod(
    spiral_n_elem,
    spiral_start,
    y_direction,
    normal,
    spiral_length,
    base_radius=radius_along_spiral,
    density=density,
    # nu=damping,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
    position = spiral_position
)

# simulator.append(
#     spiral_thread
# )  # Now rod is ready for simulation, append rod to simulation

# simulator.dampen(spiral_thread).using(
#     AnalyticalLinearDamper, 
#     damping_constant = damping_constant_spiral,
#     time_step = sim_dt
# )

# spiral_pos_connect_idx = [50, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600]

# spiral_pos_connect_idx = tuple(spiral_pos_connect_idx)
# simulator.constrain(spiral_thread).using(
#     FixedConstraint, constrained_position_idx=spiral_pos_connect_idx, constrained_director_idx=spiral_pos_connect_idx
# )

# j = 0
# for i in range(len(spiral_pos_connect_idx)): #len(spiral_pos_connect_idx)

#     if j > 2:
#         j = 0

#     simulator.connect(first_rod=radial_thread[j], second_rod=spiral_thread,
#                       first_connect_idx=rad_thread_connect_idx[i], second_connect_idx=spiral_pos_connect_idx[i]).using(FixedJoint,
#                                                                                                                         k=0.0,
#                                                                                                                         nu=0.0,
#                                                                                                                         kt=0.0)
#     j = j + 1

if add_ext_forcing:
    for i in range(int(num_rad_threads/2)):
        simulator.add_forcing_to(radial_thread[i]).using(NoForces)
    simulator.add_forcing_to(boundary_thread).using(NoForces)
    simulator.add_forcing_to(spiral_thread).using(NoForces)

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
                    return
                
    # Collect data using callback function for postprocessing
    post_processing_dict_radial_thread = []
    for i in range(int(num_rad_threads/2)):
        post_processing_dict_radial_thread_each = defaultdict(list)
        simulator.collect_diagnostics(radial_thread[i]).using(
            WebCallBack,
            step_skip=step_skip,
            callback_params=post_processing_dict_radial_thread_each,
        )

        post_processing_dict_radial_thread.append(post_processing_dict_radial_thread_each)

        # post_processing_dict_radial_thread[f"thread {i+1}"].append(post_processing_dict_radial_thread_each)

    post_processing_dict_boundary_thread = defaultdict(list)
    simulator.collect_diagnostics(boundary_thread).using(
        WebCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_boundary_thread
    )

    # post_processing_dict_spiral_thread = defaultdict(list)
    # simulator.collect_diagnostics(spiral_thread).using(
    #     WebCallBack,
    #     step_skip=step_skip,
    #     callback_params = post_processing_dict_spiral_thread
    # )

simulator.finalize()

current_time = 0.0

maxwidth = 800
aspect_ratio = 3 / 4

viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)

renderer = Session(width=maxwidth, height=int(maxwidth*aspect_ratio))

bending_energy_0 = []
bending_energy_1 = []
bending_energy_2 = []
bending_energy_bound = []

shear_energy_0 = []
shear_energy_1 = []
shear_energy_2 = []
shear_energy_bound = []

translational_energy_0 = []
translational_energy_1 = []
translational_energy_2 = []
translational_energy_bound = []

rotational_energy_0 = []
rotational_energy_1 = []
rotational_energy_2 = []
rotational_energy_bound = []

first_interval_time = update_interval + current_time
last_interval_time = current_time + max_episode_final_time
for stop_time in np.arange(
    first_interval_time, last_interval_time + sim_dt, update_interval
):
    current_time = run_and_update_plot(simulator, StatefulStepper, sim_dt, current_time, stop_time, viewer, renderer)
    bending_energy_0.append(CosseratRod.compute_bending_energy(radial_thread[0]))
    shear_energy_0.append(CosseratRod.compute_shear_energy(radial_thread[0]))
    translational_energy_0.append(CosseratRod.compute_translational_energy(radial_thread[0]))
    rotational_energy_0.append(CosseratRod.compute_rotational_energy(radial_thread[0]))

    bending_energy_1.append(CosseratRod.compute_bending_energy(radial_thread[1]))
    shear_energy_1.append(CosseratRod.compute_shear_energy(radial_thread[1]))
    translational_energy_1.append(CosseratRod.compute_translational_energy(radial_thread[1]))
    rotational_energy_1.append(CosseratRod.compute_rotational_energy(radial_thread[1]))

    bending_energy_2.append(CosseratRod.compute_bending_energy(radial_thread[2]))
    shear_energy_2.append(CosseratRod.compute_shear_energy(radial_thread[2]))
    translational_energy_2.append(CosseratRod.compute_translational_energy(radial_thread[2]))
    rotational_energy_2.append(CosseratRod.compute_rotational_energy(radial_thread[2]))

    bending_energy_bound.append(CosseratRod.compute_bending_energy(boundary_thread))
    shear_energy_bound.append(CosseratRod.compute_shear_energy(boundary_thread))
    translational_energy_bound.append(CosseratRod.compute_translational_energy(boundary_thread))
    rotational_energy_bound.append(CosseratRod.compute_rotational_energy(boundary_thread))
    # print(radial_thread[0].position_collection)
    # print(radial_thread[1].position_collection)
    # print(radial_thread[2].position_collection)
    # print(CosseratRod.compute_rotational_energy(radial_thread[0]))
    print(CosseratRod.compute_bending_energy(boundary_thread))
    # print(CosseratRod.compute_rotational_energy(radial_thread[2]))
    # print("------------")

renderer.close()
viewer.close()

rods_history = [post_processing_dict_radial_thread[i] for i in range(len(post_processing_dict_radial_thread))]
rods_history.append(post_processing_dict_boundary_thread)
video_name = "web_video_k1_D20k.mp4"
plot_web_video(rods_history=rods_history, video_name=video_name, fps=rendering_fps)

for i in range(int(num_rad_threads/2)):
    if i == 0:
        bending_energy_list = bending_energy_0
        shear_energy_list = shear_energy_0
        translational_energy_list = translational_energy_0
        rotational_energy_list = rotational_energy_0
    elif i == 1:
        bending_energy_list = bending_energy_1
        shear_energy_list = shear_energy_1
        translational_energy_list = translational_energy_1
        rotational_energy_list = rotational_energy_1
    else:
        bending_energy_list = bending_energy_2
        shear_energy_list = shear_energy_2
        translational_energy_list = translational_energy_2
        rotational_energy_list = rotational_energy_2

    post_processing_dict_radial_thread_each_temp = post_processing_dict_radial_thread[0]

    position = np.array(post_processing_dict_radial_thread_each_temp["position"])
    velocity = np.array(post_processing_dict_radial_thread_each_temp["velocity_collection"])
    acceleration = np.array(post_processing_dict_radial_thread_each_temp["acceleration_collection"])
    forces = np.array(post_processing_dict_radial_thread_each_temp["internal_forces"])
    torques = np.array(post_processing_dict_radial_thread_each_temp["internal_torques"])
    bending_energy = np.array(bending_energy_list)
    shear_energy = np.array(shear_energy_list)
    translational_energy = np.array(translational_energy_list)
    rotational_energy = np.array(rotational_energy_list)
    tangents = np.array(post_processing_dict_radial_thread_each_temp["tangents"])
    kappa = np.array(post_processing_dict_radial_thread_each_temp["kappa"])
    omega = np.array(post_processing_dict_radial_thread_each_temp["omega_collection"])
    sigma = np.array(post_processing_dict_radial_thread_each_temp["sigma"])

    np.savez(
        file = f"radial_thread_{i}",
        position = position,
        velocity = velocity,
        acceleration = acceleration,
        forces = forces,
        torques = torques,
        bending_energy = bending_energy,
        shear_energy = shear_energy,
        translational_energy = translational_energy,
        rotational_energy = rotational_energy,
        tangents = tangents,
        kappa = kappa,
        omega = omega,
        sigma = sigma
    )

position = np.array(post_processing_dict_boundary_thread["position"])
velocity = np.array(post_processing_dict_boundary_thread["velocity_collection"])
acceleration = np.array(post_processing_dict_boundary_thread["acceleration_collection"])
forces = np.array(post_processing_dict_boundary_thread["internal_forces"])
torques = np.array(post_processing_dict_boundary_thread["internal_torques"])
bending_energy = np.array(bending_energy_bound)
shear_energy = np.array(shear_energy_bound)
translational_energy = np.array(translational_energy_bound)
rotational_energy = np.array(rotational_energy_bound)
tangents = np.array(post_processing_dict_boundary_thread["tangents"])
kappa = np.array(post_processing_dict_boundary_thread["kappa"])
omega = np.array(post_processing_dict_boundary_thread["omega_collection"])
sigma = np.array(post_processing_dict_boundary_thread["sigma"])

np.savez(
    file = "boundary_thread",
    position = position,
    velocity = velocity,
    acceleration = acceleration,
    forces = forces,
    torques = torques,
    bending_energy = bending_energy,
    shear_energy = shear_energy,
    translational_energy = translational_energy,
    rotational_energy = rotational_energy,
    tangents = tangents,
    kappa = kappa,
    omega = omega,
    sigma = sigma
)