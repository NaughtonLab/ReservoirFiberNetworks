#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# read_write.py
#
#  Created on: 2016. 6. 16.
#      Author: Ryu Woon Jung (Leon)
#

#
# *********     Read and Write Example      *********
#
#
# Available DXL model on this example : All models using Protocol 1.0
# This example is designed for using a Dynamixel MX-28, and an USB2DYNAMIXEL.
# To use another Dynamixel model, such as X series, see their details in E-Manual(support.robotis.com) and edit below variables yourself.
# Be sure that Dynamixel MX properties are already set as ## ID : 1 / Baudnum : 1 (Baudrate : 1000000 [1M])
#

import os, sys
import numpy as np
import pandas as pd
import time
from scipy.interpolate import CubicSpline

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    def getch():
        return sys.stdin.read(1)

# os.sys.path.append('../dynamixel_functions_py')             # Path setting

import dynamixel_sdk as dynamixel                    # Uses DYNAMIXEL SDK library

# Control table address
ADDR_MX_TORQUE_ENABLE       = 24                            # Control table address is different in Dynamixel model
ADDR_MX_GOAL_POSITION       = 30
ADDR_MX_PRESENT_POSITION    = 36

# Protocol version
PROTOCOL_VERSION            = 1                             # See which protocol version is used in the Dynamixel

# Default setting
DXL_ID                      = 1                             # Dynamixel ID: 1
BAUDRATE                    = 1000000
DEVICENAME                  = "COM4"                        # Check which port is being used on your controller
                                                            # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0"

TORQUE_ENABLE               = 1                             # Value for enabling the torque
TORQUE_DISABLE              = 0                             # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 0                             # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 1023                          # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 10                            # Dynamixel moving status threshold

ESC_ASCII_VALUE             = 0x1b

COMM_SUCCESS                = 0                             # Communication Success result value
COMM_TX_FAIL                = -1001                         # Communication Tx Failed

# Initialize PortHandler Structs
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
port_num = dynamixel.PortHandler(DEVICENAME)

# Initialize PacketHandler Structs
protocol_packet_handler = dynamixel.PacketHandler(PROTOCOL_VERSION)

# Load simulation results
sim_spacing_desired = 100  # in mm
exp_radius = 70.0  # in mm
network_size = 6
exp_spacing = 520 / (network_size + 1)  # in mm
file_name = f"Experiments\SAGE\delta_for_SimSpacing{sim_spacing_desired}mm_Exp{network_size}by{network_size}.csv"
params = pd.read_csv(file_name)
theta_range_list = params['Experimental Theta (deg)'].dropna().round(2).drop_duplicates().tolist()
theta_range_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
#  rho = 1
# exp_delta = exp_spacing
# theta_rho_1 = np.arcsin(exp_delta / exp_radius)  # in radians
# theta_rho_1_deg = np.degrees(theta_rho_1)
# theta_range_list.append(theta_rho_1_deg)
print("Theta range list (deg):", theta_range_list, "Number of elements:", len(theta_range_list))
theta_idx = input(f"Select theta index (0 to {len(theta_range_list)-1}): ")
theta_idx = int(theta_idx)
target_theta_deg = theta_range_list[theta_idx]

index = 0
dxl_comm_result = COMM_TX_FAIL                              # Communication result
angle_conversion = 1023 / 300
save = True

# Generating Spline for goal positions
seed_value = 1234
np.random.seed(seed_value)
duration = 60*2.5                                                                   # seconds
sample_freq = 5                                                                     # Hz
sample_time = np.ceil(duration).astype(int)
x_sample = np.linspace(0, sample_time, sample_time*sample_freq + 1)
y_sample = np.random.uniform(0, 1, size=(sample_time*sample_freq + 1,))
y_sample[0] = 0
y_sample[-1] = 0
spline = CubicSpline(x_sample, y_sample)

# Generating goal positions for target theta
mean = 240 # degrees
rng = target_theta_deg
min_angle = mean #np.rint((mean+rng) * angle_conversion).astype(int)
max_angle = mean - rng #np.rint(mean * angle_conversion).astype(int)
angle_limit = 300 # degrees
goal_positions_deg = spline(x_sample) * rng + mean
goal_positions_deg = np.clip(goal_positions_deg, 0, angle_limit)
dxl_goal_position = [int(pos * angle_conversion) for pos in goal_positions_deg]           # Goal position
# print("Generated goal positions:", dxl_goal_position)

dxl_error = 0                                               # Dynamixel error
dxl_present_position = 0                                    # Present position

def read_present_position(max_retries=3, sleep_s=0.005):
    for _ in range(max_retries):
        try:
            pos, comm, err = protocol_packet_handler.read2ByteTxRx(
                port_num, DXL_ID, ADDR_MX_PRESENT_POSITION
            )
        except IndexError:
            # Short/empty packet – back off and retry
            time.sleep(sleep_s)
            continue

        if comm != COMM_SUCCESS:
            # Print and retry
            print(protocol_packet_handler.getTxRxResult(comm))
            time.sleep(sleep_s)
            continue
        if err != 0:
            print(protocol_packet_handler.getRxPacketError(err))
            time.sleep(sleep_s)
            continue

        return pos  # success
    return None  # exhausted

# Open port
if port_num.openPort():
    print("Succeeded to open the port!")
else:
    print("Failed to open the port!")
    print("Press any key to terminate...")
    msvcrt.getch()
    quit()

# Set port baudrate
if port_num.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate!")
else:
    print("Failed to change the baudrate!")
    print("Press any key to terminate...")
    msvcrt.getch()
    quit()


# Enable Dynamixel Torque
dxl_comm_result, dxl_error = protocol_packet_handler.write1ByteTxRx(port_num, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print(protocol_packet_handler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(protocol_packet_handler.getRxPacketError(dxl_error))
else:
    print("Dynamixel has been successfully connected")

store_data = {'Time': [], 'GoalPos': [], 'PresPos': []}
cycle = 0

print("Start the camera...")
time.sleep(10)
print("Starting experiment...")

start_time = time.time()
current_goal_position = int(min_angle * angle_conversion)
print(f"Initial goal position: {current_goal_position} ticks")

while (time.time() - start_time) <= duration:
    # print("Press any key to continue! (or press ESC to quit!)")
    # if msvcrt.getch().decode() == chr(ESC_ASCII_VALUE):
    #     break

    # Write goal position
    # print(f"Moving to goal position: {current_goal_position} ticks")
    dxl_comm_result, dxl_error = protocol_packet_handler.write2ByteTxRx(port_num, DXL_ID, ADDR_MX_GOAL_POSITION, current_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print(protocol_packet_handler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(protocol_packet_handler.getRxPacketError(dxl_error))

    while True:
        dxl_present_position = read_present_position()
        if dxl_present_position is None:
            # Couldn’t get a clean read; decide whether to continue or abort
            continue

        store_data['Time'].append(time.time() - start_time)
        store_data['GoalPos'].append(current_goal_position)
        store_data['PresPos'].append(dxl_present_position)

        if abs(current_goal_position - dxl_present_position) <= DXL_MOVING_STATUS_THRESHOLD:
            cycle += 1
            next_angle = spline(time.time() - start_time) * rng + mean
            next_angle = np.clip(next_angle, 0, angle_limit)
            current_goal_position = int(next_angle * angle_conversion) #dxl_goal_position[index] #
            break

        # time.sleep(0.003)  # tiny poll delay
    
    # time.sleep(0.25)

    # Change goal position
    if index < len(dxl_goal_position) - 1:
        index = index + 1
    else:
        index = 0

print("Bringing Dynamixel back to mean position...")

## Bring dynamixel back to mean position
# Write goal position
mean_position = np.rint(mean * angle_conversion).astype(int)
dxl_comm_result, dxl_error = protocol_packet_handler.write2ByteTxRx(port_num, DXL_ID, ADDR_MX_GOAL_POSITION, mean_position)
if dxl_comm_result != COMM_SUCCESS:
    print(protocol_packet_handler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(protocol_packet_handler.getRxPacketError(dxl_error))

while True:
    dxl_present_position = read_present_position()
    if dxl_present_position is None:
        # Couldn’t get a clean read; decide whether to continue or abort
        continue

    store_data['Time'].append(time.time() - start_time)
    store_data['GoalPos'].append(current_goal_position)
    store_data['PresPos'].append(dxl_present_position)

    if not (abs(mean_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD):
        break

# Disable Dynamixel Torque
dxl_comm_result, dxl_error = protocol_packet_handler.write1ByteTxRx(port_num, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print(protocol_packet_handler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(protocol_packet_handler.getRxPacketError(dxl_error))

# Save data
data = [store_data]
if save:
    np.savez(f'Experiments/SAGE/{network_size}by{network_size}/ForceSpacingExp_{mean}_{rng}_Target{target_theta_deg}deg_{network_size}by{network_size}_{sample_freq}Hz_OldCode.npz', data=data)
print("Experiment data saved.")

# Close port
port_num.closePort()