from typing import Optional, Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
import matplotlib.animation as manimation

from abc import ABC, abstractmethod

from gym_fiber import RENDERER_CONFIG
from gym_fiber.config import RendererType
from gym_fiber.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)

import pkg_resources

def render_figure(fig:plt.figure):
    w, h = fig.get_size_inches()
    dpi_res = fig.get_dpi()
    w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h*dpi_res))

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    data = np.asarray(canvas.buffer_rgba())[:,:,:3]
    return data

def convert_marker_size(radius, ax):
    """
    Convert marker size from radius to s (in scatter plot).

    Parameters
    ----------
    radius : np.array or float
        Array (or a number) of radius
    ax : matplotlib.Axes
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    max_axis_length = max(abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0]))
    scaling_factor = 3.0e3 * (2*0.1) / max_axis_length

    return np.sqrt(np.pi * (scaling_factor * radius))
    #ppi = 72 # standard point size in matplotlib is 72 points per inch (ppi), no matter the dpi
    #point_whole_ax = 5 * 0.8 * ppi
    #point_radius= 2 * radius / 1.0 * point_whole_ax
    #return point_radius**2

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class Geom(ABC):
    @abstractmethod
    def __call__(self):
        pass


class ElasticaRod(Geom):
    # RGB color must be 2d array 
    rgb_color = np.array([[0.35, 0.29, 1.0]])

    def __init__(self, position, radius, ax):
        # self.rod = rod
        self.ax = ax

        # Initialize scatter plot
        # pos, rad = self.get_position_radius()
        self.pos = position
        self.rad = radius / 2
        if not self.pos.shape[-1] == self.rad.shape[0]:
            # radius defined at element, while position defined at node.
            # typical elastica has n_node = n_elem + 1 (unless the rod is circular)
            self.pos = 0.5 * (self.pos[..., 1:] + self.pos[..., :-1])

        marker_size = convert_marker_size(self.rad, self.ax)
        # print(f"before plotting len(pos) = {self.pos.shape}, len(rad) = {len(self.rad)}, len(marker)={len(marker_size)}")
        self.scatter = ax.scatter(self.pos[0,:], self.pos[1,:], self.pos[2,:], s=marker_size, c=ElasticaRod.rgb_color)

    def __call__(self):
        # Update scatter plot positions
        # pos, rad = self.get_position_radius()
        self.scatter._offsets3d = tuple(self.pos)

        # Updater radius
        self.scatter.set_sizes(convert_marker_size(self.rad, self.ax))
        
        return self.scatter

class ElasticaRodDirector(Geom):
    # TODO
    def __init__(self, rod, ax):
        self.ax = ax

    def __call__(self):
        return None


class ElasticaSphere(Geom):
    rgb_color = np.array([1.0, 0.0, 1.0])

    def __init__(self, loc, radius, ax):
        # Initialize scatter plot
        self.scatter = ax.scatter(loc[0], loc[1], loc[2], s=convert_marker_size(radius, ax), c=ElasticaSphere.rgb_color)

    def __call__(self):
        return self.scatter


class Session(BaseElasticaRendererSession, BaseRenderer):
    def __init__(self, width, height, dpi=100):
        self.object_collection = []
        self.width = width
        self.height = height
        self.dpi = dpi

        px = 1.0 / dpi
        self.fig = plt.figure(
            figsize=(width*px,height*px),
            frameon=True,
            dpi=dpi,
        )
        self.ax = plt.axes(projection="3d")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    @property
    def type(self):
        return RendererType.MATPLOTLIB

    def add_rod(self, data):
        position = data[0]
        radius = data[1]
        self.object_collection.append(ElasticaRod(position, radius, self.ax))
        # TODO Maybe give another configuration to plot the directors
        # self.object_collection.append(ElasticaRodDirector(rod, self.ax))

    def add_rigid_body(self, body):
        self.object_collection.append(ElasticaCylinder(body, self.ax))

    def add_point(self, loc: list, radius: float):
        # Add static sphere
        self.object_collection.append(ElasticaSphere(loc, radius, self.ax))

    def render(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_param: Optional[tuple] = None, # POVray parameter
        **kwargs
    ):
        # Reset width and height
        if not width:
            width = self.width
        if not height:
            height = self.height

        # Maybe convert povray cmaera_param to matplotlib viewpoint

        objects = [obj() for obj in self.object_collection]
        self.rescale_axis()
        rendered_data = render_figure(self.fig)

        return rendered_data

    def close(self):
        plt.close(plt.gcf())
        self.object_collection.clear()

    def rescale_axis(self):
        self.ax.relim()
        self.ax.autoscale_view()
        set_axes_equal(self.ax)

class save_video():

    def __init__(self, maxwidth, aspect_ratio, target_position, sphere_radius):

        self.maxwidth = maxwidth
        self.aspect_ratio = aspect_ratio
        self.viewer = None
        self.renderer = None
        self.target_position = target_position
        self.sphere_radius = sphere_radius

    def plot_video(self, plot_params_list: list, video_name: str, margin=0.2, fps=60, plot3d=True):

        t = np.array(plot_params_list["time"])
        position = []
        radius = []

        for plot_params in plot_params_list["position"]:
            positions_over_time = np.array(plot_params)
            position.append(positions_over_time)
        
        for plot_params in plot_params_list["radius"]:
            radius_over_time = np.array(plot_params)
            radius.append(radius_over_time)
            
        total_time = t[...,-1] #int(np.around(t[..., -1], 1))
        # print(f"length of time = {len(t)}, length of pos = {len(position)}, length of radius = {len(radius)}")
        total_frames = fps * total_time
        dump = round(len(t) / total_frames)
        print("creating video -- this can take a few minutes")
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        frames = []

        dpi = 100
        px = 1.0 / dpi
        fig = plt.figure(
            figsize=(self.maxwidth*px,self.maxwidth*self.aspect_ratio*px),
            frameon=True,
            dpi=dpi)

        for time in range(1, len(t), dump):
            if self.renderer is None:
                # print(f"inside for loop = {len(radius[time])}")
                self.renderer = Session(width=self.maxwidth, height=int(self.maxwidth*self.aspect_ratio))
                self.renderer.add_rod([position[time], radius[time]])  
                self.renderer.add_point(self.target_position, self.sphere_radius)

            # POVRAY
            if RENDERER_CONFIG == RendererType.MATPLOTLIB:
                state_image = self.renderer.render()
            else:
                raise NotImplementedError("Rendering module is not imported properly")

            frames.append(state_image)

        def animate(i):
            if self.viewer is None:
                from gym_fiber.utils.render import pyglet_rendering
                self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=self.maxwidth)
            plt.axis('on')
            plt.imshow(frames[i])

        anim = manimation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(video_name, writer="imagemagick")

        self.viewer.close()
        # self.renderer.close()

        print("Animation saved")
