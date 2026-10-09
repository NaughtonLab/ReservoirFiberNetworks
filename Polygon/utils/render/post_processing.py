import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


def plot_network_video(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs
):
    plt.rcParams.update({"font.size": 17})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    # rods_history = kwargs.get("rods_history")

    # simulation time
    sim_time = np.array(rods_history[0]["time"])
    # print(sim_time)

    # Rods
    n_visualized_rods = len(rods_history) #kwargs.get("n_visualized_rods") 
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-55.0, 55.0))
    ylim = kwargs.get("y_limits", (-55.0, 55.0))
    zlim = kwargs.get("z_limits", (-1.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(abs(difference(xlim)), abs(difference(ylim)))
    # The scaling factor from physical space to matplotlib space
    # scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    # scaling_factor *= 2.6e3  # Along one-axis
    scaling_factor = 3.0e3 * (2*0.1) / max_axis_length

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection="3d")
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(elev=90, azim=270, roll=0)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)

        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
    
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]
        
        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            inst_position[2],
            s=np.sqrt(np.pi * (scaling_factor * inst_radius[0]))
        )

    # ax.set_aspect("equal")
    video_name = video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    # print(inst_position)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )
                    # print(inst_position)

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])
                    rod_lines[rod_idx].set_zdata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[1])
                    rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[0],
                        inst_position[1],
                        inst_position[2],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.sqrt(np.pi * (scaling_factor * inst_radius))
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_network_video_2D(
    rods_history: Sequence[Dict],
    # sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info

    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
        rods_history[rod_idx]["directors"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]

    # plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
    
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)


    xlim = kwargs.get("x_limits", (-10, 110))
    ylim = kwargs.get("y_limits", (-3, 3))
    # zlim = kwargs.get("z_limits", (-0.05*500, 1.0*500))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3 * 2  # Along one-axis

    fig = plt.figure(2, figsize=(20, 16), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    
    time_idx = 0
    rod_nodes = [None for _ in range(n_visualized_rods)]
    rod_lines = [None for _ in range(n_visualized_rods)]
    radius_upper = [None for _ in range(n_visualized_rods)]
    radius_lower = [None for _ in range(n_visualized_rods)]
    # rod_com_lines = [None for _ in range(n_visualized_rods)]
    # rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius, inst_directors = rod_history_unpacker(rod_idx, time_idx)

        rod_nodes[rod_idx] = ax.plot(inst_position[0], inst_position[1], "o")[0]

        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=2.0, label='time: %.3f' % (np.round(sim_time[time_idx],3)))[0]

        # print(inst_directors[0])

        radius_upper[rod_idx] = ax.plot(
            inst_position[0] + inst_directors[0][0] * inst_radius, 
            inst_position[1] + inst_directors[0][1] * inst_radius, 
            "b", lw=2.0, )[0]
        
        radius_lower[rod_idx] = ax.plot(
            inst_position[0] - inst_directors[0][0] * inst_radius, 
            inst_position[1] - inst_directors[0][1] * inst_radius, 
            "b", lw=2.0, )[0]


    # ax.legend()
    ax.grid(True)

    # ax.set_aspect("equal")
    video_name = video_name
    params_str = kwargs.get("params_str", "0")

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                # print(sim_time[time_idx])
                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius, inst_directors = rod_history_unpacker(rod_idx, time_idx)
                    rod_nodes[rod_idx].set_xdata(inst_position[0])
                    rod_nodes[rod_idx].set_ydata(inst_position[1])
                    

                    rod_lines[rod_idx].set_label('time: %.3f' % (np.round(sim_time[time_idx],3)) )

                    inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])


                    radius_upper[rod_idx].set_xdata(inst_position[0] + inst_directors[0][0] * inst_radius)
                    radius_upper[rod_idx].set_ydata(inst_position[1] + inst_directors[0][1] * inst_radius)


                    radius_lower[rod_idx].set_xdata(inst_position[0] - inst_directors[0][0] * inst_radius)
                    radius_lower[rod_idx].set_ydata(inst_position[1] - inst_directors[0][1] * inst_radius)

                # ax.legend()
                ax.set_title(params_str)

                writer.grab_frame()


def plot_network_video_2D_less_callback(
    rods_history: Sequence[Dict],
    # sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info

    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        # rods_history[rod_idx]["radius"][t_idx],
        # rods_history[rod_idx]["directors"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]

    # plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
    
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)


    xlim = kwargs.get("x_limits", (-10, 110))
    ylim = kwargs.get("y_limits", (-3, 3))
    # zlim = kwargs.get("z_limits", (-0.05*500, 1.0*500))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3 * 2  # Along one-axis

    fig = plt.figure(2, figsize=(20, 16), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    
    time_idx = 0
    rod_nodes = [None for _ in range(n_visualized_rods)]
    rod_lines = [None for _ in range(n_visualized_rods)]
    # radius_upper = [None for _ in range(n_visualized_rods)]
    # radius_lower = [None for _ in range(n_visualized_rods)]
    # rod_com_lines = [None for _ in range(n_visualized_rods)]
    # rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        # inst_position, inst_radius, inst_directors = rod_history_unpacker(rod_idx, time_idx)
        inst_position = rod_history_unpacker(rod_idx, time_idx)
        inst_position = inst_position[0]

        rod_nodes[rod_idx] = ax.plot(inst_position[0], inst_position[1], "o")[0]

        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=2.0, label='time: %.3f' % (np.round(sim_time[time_idx],3)))[0]

        # print(inst_directors[0])

        # radius_upper[rod_idx] = ax.plot(
        #     inst_position[0] + inst_directors[0][0] * inst_radius, 
        #     inst_position[1] + inst_directors[0][1] * inst_radius, 
        #     "b", lw=2.0, )[0]
        
        # radius_lower[rod_idx] = ax.plot(
        #     inst_position[0] - inst_directors[0][0] * inst_radius, 
        #     inst_position[1] - inst_directors[0][1] * inst_radius, 
        #     "b", lw=2.0, )[0]


    # ax.legend()
    ax.grid(True)

    # ax.set_aspect("equal")
    video_name = video_name
    params_str = kwargs.get("params_str", "0")

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                # print(sim_time[time_idx])
                for rod_idx in range(n_visualized_rods):
                    # inst_position, inst_radius, inst_directors = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = inst_position[0]
                    rod_nodes[rod_idx].set_xdata(inst_position[0])
                    rod_nodes[rod_idx].set_ydata(inst_position[1])
                    

                    rod_lines[rod_idx].set_label('time: %.3f' % (np.round(sim_time[time_idx],3)) )

                    inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])


                    # radius_upper[rod_idx].set_xdata(inst_position[0] + inst_directors[0][0] * inst_radius)
                    # radius_upper[rod_idx].set_ydata(inst_position[1] + inst_directors[0][1] * inst_radius)


                    # radius_lower[rod_idx].set_xdata(inst_position[0] - inst_directors[0][0] * inst_radius)
                    # radius_lower[rod_idx].set_ydata(inst_position[1] - inst_directors[0][1] * inst_radius)

                # ax.legend()
                ax.set_title(params_str)

                writer.grab_frame()
