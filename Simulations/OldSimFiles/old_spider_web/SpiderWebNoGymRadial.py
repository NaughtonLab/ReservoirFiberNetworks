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

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass

def render_web(radial_thread, num_rad_threads, viewer, renderer):

    renderer.add_rod(radial_thread, 'rad')
    # renderer.add_rod(radial_thread[1], 'rad')
    # renderer.add_rod(radial_thread[2], 'rad')
    # for i in range(num_boundary_threads):
    # renderer.add_rod(boundary_thread, 'bound')
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
    render_web(radial_thread, num_rad_threads, viewer, renderer)
    return time

'''Web geometry'''
# The number of radial threads should be the number of lines starting from origin and extending till the web boundary
# Thus for a hexagonal web the number of radial threads will be 6
# Stick with even number of radial threads for now
num_rad_threads = 6
num_boundary_threads = num_rad_threads
web_origin = np.zeros((3,))
web_radius = 1000/10
web_plane = 'xy'
boundary_length = 1000*num_boundary_threads
# If web_radius reduces by a factor of n then reduce the spiral_offset and spiral_param_a by the same factor n to keep the geometry constant
spiral_offset = 100/10
spiral_param_a = 22/10
spiral_n_turns = 6

'''number of elements in each type of thread'''
dia_n_elem = 120 # This number indicates the number of elements in the diameter of the web. Half of these + 1 will be the number of elements in each side of the polygon.
spiral_n_elem_first_half = 50
spiral_n_elem_second_half = 100
spiral_n_elem = (spiral_n_elem_first_half + spiral_n_elem_second_half) * int(spiral_n_turns/2) * num_rad_threads

'''Physical simulation timestep in seconds'''
sim_dt = 1e-5 #or 1e-4

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
radius = 5.0 / 10                     

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

density = 1300 * 1e-6
radius_tip = radius  # radius of the arm at the tip
radius_base = radius_tip  # radius of the arm at the base

radius_along_diameter = np.linspace(radius_tip, radius_tip, dia_n_elem)
dx_dia = 2*web_radius/dia_n_elem
damping_constant_rad = 5*1e3

ang = 1 * angle_between_threads
direction = np.array([np.cos(ang), np.sin(ang), 0.0])
start = web_origin - web_radius * direction

start_direc_vec = web_origin - web_radius * direction
end_direc_vec = web_origin + web_radius * direction
direction_position = np.linspace(start_direc_vec, end_direc_vec, dia_n_elem+1)
direction_position = direction_position.T

radial_thread = CosseratRod.straight_rod(
    dia_n_elem,
    start,
    direction,
    normal,
    base_length=2*web_radius,
    base_radius=radius_along_diameter,
    density=density,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
    # position = direction_position
    )

simulator.append(
    radial_thread
)

# simulator.dampen(radial_thread).using(
#     AnalyticalLinearDamper, 
#     damping_constant = damping_constant_rad,
#     time_step = sim_dt
# )

# simulator.constrain(radial_thread).using(
#     FixedConstraint, constrained_position_idx=(0,-1), constrained_director_idx=(0,-1)
# )

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
    post_processing_dict_radial_thread = defaultdict(list)
    simulator.collect_diagnostics(radial_thread).using(
        WebCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_radial_thread,
    )


simulator.finalize()

current_time = 0.0

maxwidth = 800
aspect_ratio = 3 / 4

viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)

renderer = Session(width=maxwidth, height=int(maxwidth*aspect_ratio))

bending_energy_0 = []
shear_energy_0 = []
translational_energy_0 = []
rotational_energy_0 = []

first_interval_time = update_interval + current_time
last_interval_time = current_time + max_episode_final_time
for stop_time in np.arange(
    first_interval_time, last_interval_time + sim_dt, update_interval
):
    current_time = run_and_update_plot(simulator, StatefulStepper, sim_dt, current_time, stop_time, viewer, renderer)
    bending_energy_0.append(CosseratRod.compute_bending_energy(radial_thread))
    shear_energy_0.append(CosseratRod.compute_shear_energy(radial_thread))
    translational_energy_0.append(CosseratRod.compute_translational_energy(radial_thread))
    rotational_energy_0.append(CosseratRod.compute_rotational_energy(radial_thread))

    print(CosseratRod.compute_bending_energy(radial_thread))
    print("------------")

renderer.close()
viewer.close()


position = np.array(post_processing_dict_radial_thread["position"])
velocity = np.array(post_processing_dict_radial_thread["velocity_collection"])
acceleration = np.array(post_processing_dict_radial_thread["acceleration_collection"])
forces = np.array(post_processing_dict_radial_thread["internal_forces"])
torques = np.array(post_processing_dict_radial_thread["internal_torques"])
bending_energy = np.array(bending_energy_0)
shear_energy = np.array(shear_energy_0)
translational_energy = np.array(translational_energy_0)
rotational_energy = np.array(rotational_energy_0)
tangents = np.array(post_processing_dict_radial_thread["tangents"])
kappa = np.array(post_processing_dict_radial_thread["kappa"])
omega = np.array(post_processing_dict_radial_thread["omega_collection"])
sigma = np.array(post_processing_dict_radial_thread["sigma"])

np.savez(
    file = f"radial_thread",
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

