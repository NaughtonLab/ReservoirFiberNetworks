import os, sys, time, csv, subprocess, tempfile, pathlib, threading
from string import Template
from dataclasses import dataclass
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import dynamixel_sdk as dynamixel                    # Uses DYNAMIXEL SDK library

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

class MotorControl:
    def __init__(self, config):
        self.config = config
        # Initialize PortHandler Structs
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.port_num = dynamixel.PortHandler(self.config.DEVICENAME)

        # Initialize PacketHandler Structs
        self.protocol_packet_handler = dynamixel.PacketHandler(self.config.PROTOCOL_VERSION)

        self.dxl_comm_result = dynamixel.COMM_TX_FAIL                    # Communication result
        self.angle_conversion = 1023 / 300
        self.dxl_error = 0                                               # Dynamixel error
        self.dxl_present_position = 0                                    # Present position

    def clear_port(self):
        try:
            self.port_num.clearPort()
        except Exception:
            pass
    
    def check_port_and_baud(self):
        # Open port
        if self.port_num.openPort():
            print("Succeeded to open the port!")
        else:
            print("Failed to open the port!")
            print("Press any key to terminate...")
            msvcrt.getch()
            quit()

        # Set port baudrate
        if self.port_num.setBaudRate(self.config.BAUDRATE):
            print("Succeeded to change the baudrate!")
        else:
            print("Failed to change the baudrate!")
            print("Press any key to terminate...")
            msvcrt.getch()
            quit()

    def enable_torque(self):
        # Enable Dynamixel Torque
        self.dxl_comm_result, self.dxl_error = self.protocol_packet_handler.write1ByteTxRx(self.port_num, self.config.DXL_ID, self.config.ADDR_MX_TORQUE_ENABLE, self.config.TORQUE_ENABLE)
        if self.dxl_comm_result != self.config.COMM_SUCCESS:
            print(self.protocol_packet_handler.getTxRxResult(self.dxl_comm_result))
        elif self.dxl_error != 0:
            print(self.protocol_packet_handler.getRxPacketError(self.dxl_error))
        else:
            print("Dynamixel has been successfully connected")

    def disable_torque(self):
        # Disable Dynamixel Torque
        for attempt in range(3):
            self.clear_port()
            self.dxl_comm_result, self.dxl_error = self.protocol_packet_handler.write1ByteTxRx(self.port_num, self.config.DXL_ID, self.config.ADDR_MX_TORQUE_ENABLE, self.config.TORQUE_DISABLE)
            if self.dxl_comm_result == self.config.COMM_SUCCESS and self.dxl_error == 0:
                return True
            if attempt < 2:
                time.sleep(0.05)
        if self.dxl_comm_result != self.config.COMM_SUCCESS:
            print(self.protocol_packet_handler.getTxRxResult(self.dxl_comm_result))
        elif self.dxl_error != 0:
            print(self.protocol_packet_handler.getRxPacketError(self.dxl_error))
        return False

    def connect_and_enable_motor(self):
        self.check_port_and_baud()
        self.enable_torque()

    def set_moving_speed_rpm(self, rpm):
        # AX joint mode: value * 0.111 rpm
        value = int(np.clip(rpm / 0.111, 1, 1023))
        comm, err = self.protocol_packet_handler.write2ByteTxRx(
            self.port_num,
            self.config.DXL_ID,
            self.config.ADDR_MX_MOVING_SPEED,
            value
        )
        return comm == self.config.COMM_SUCCESS and err == 0
    
    def read_present_position(self, max_retries=3, sleep_s=0.005):
        for _ in range(max_retries):
            try:
                pos, comm, err = self.protocol_packet_handler.read2ByteTxRx(
                    self.port_num, self.config.DXL_ID, self.config.ADDR_MX_PRESENT_POSITION
                )
            except IndexError:
                # Short/empty packet – back off and retry
                time.sleep(sleep_s)
                continue

            if comm != self.config.COMM_SUCCESS:
                # Print and retry
                print(self.protocol_packet_handler.getTxRxResult(comm))
                time.sleep(sleep_s)
                continue
            if err != 0:
                print(self.protocol_packet_handler.getRxPacketError(err))
                time.sleep(sleep_s)
                continue

            return pos  # success
        return None  # exhausted
    
    def write_goal_position(self, goal_pos):
        self.dxl_comm_result, self.dxl_error = self.protocol_packet_handler.write2ByteTxRx(self.port_num, self.config.DXL_ID, self.config.ADDR_MX_GOAL_POSITION, goal_pos)
        if self.dxl_comm_result != self.config.COMM_SUCCESS:
            print(self.protocol_packet_handler.getTxRxResult(self.dxl_comm_result))
        elif self.dxl_error != 0:
            print(self.protocol_packet_handler.getRxPacketError(self.dxl_error))
        elif self.dxl_comm_result == self.config.COMM_SUCCESS and self.dxl_error == 0:
            return True
        return False
    
    def generate_goal_positions(self, amplitude_deg, mean_position_deg=205, duration=60, sample_freq=5, seed=1234, eval_freq=100):
        # Generating Spline for goal positions
        seed_value = seed
        np.random.seed(seed_value)
        sample_time = np.ceil(duration).astype(int)
        x_sample = np.linspace(0, sample_time, sample_time*sample_freq + 1)         # Coarse sampling of time points for spline generation
        y_sample = np.random.uniform(-1, 1, size=(sample_time*sample_freq + 1,))     # Random values for spline generation, will be scaled later
        y_sample[0] = 0 # start at mean position
        y_sample[-1] = 0 # end at mean position
        spline = CubicSpline(x_sample, y_sample)                                    # Create cubic spline interpolator based on random samples

        # Generating goal positions for target theta
        mean = mean_position_deg            # degrees
        min_angle = mean - amplitude_deg    # min angle in degrees
        max_angle = mean + amplitude_deg    # max angle in degrees
        angle_limit = 290                   # Physical angle limit of the motor in degrees
        eval_time_sample = np.linspace(0, sample_time, sample_time*eval_freq+1) # Fine sampling frequency for generating smooth goal positions
        goal_positions_deg = spline(eval_time_sample) * amplitude_deg + mean    # Scale spline output to desired amplitude and shift to mean position
        goal_positions_deg = np.clip(goal_positions_deg, 0, angle_limit)        # Ensure goal positions do not exceed physical limits of the motor
        dxl_goal_position = [int(pos * self.angle_conversion) for pos in goal_positions_deg]           # Goal position in Dynamixel units (0-1023 corresponding to 0-300 degrees)

        dxl_goal_position[0] = int(mean_position_deg * self.angle_conversion)  # Ensure starting at mean position
        dxl_goal_position[-1] = int(mean_position_deg * self.angle_conversion) # Ensure ending at mean position

        return dxl_goal_position
    
    def run_motor_control(self, goal_positions):
        motor_data = {'Time': [], 'GoalPos': [], 'PresPos': []}
        start_time = time.time()
        for i, goal_pos in enumerate(goal_positions):
            success = self.write_goal_position(goal_pos)
            if not success:
                print(f"Failed to write goal position {goal_pos}")
                
            while True:
                actual_position = self.read_present_position()
                if actual_position is None:
                    print(f"Failed to read present position at step {i}")
                    continue

                motor_data['Time'].append(time.time() - start_time)
                motor_data['GoalPos'].append(goal_pos)
                motor_data['PresPos'].append(actual_position)

                if abs(goal_pos - actual_position) <= self.config.DXL_MOVING_STATUS_THRESHOLD:
                    break

        return pd.DataFrame(motor_data)
    
    def run_motor_profile_and_log(self, goal_positions, start_perf, duration_s, stop_event=None, read_retries=1):
        motor_data = {'host_t_s': [], 'GoalPos': [], 'PresPos': [], 'WriteOK': []}
        if not goal_positions:
            return pd.DataFrame(motor_data)

        command_period_s = duration_s / max(len(goal_positions) - 1, 1)
        end_perf = start_perf + duration_s
        next_command_t = start_perf
        last_index = -1

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.perf_counter()
            if now >= end_perf:
                break

            while True:
                now = time.perf_counter()
                wait_s = next_command_t - now
                if wait_s <= 0 or now >= end_perf:
                    break
                time.sleep(min(wait_s, 0.001))

            elapsed_s = max(0.0, time.perf_counter() - start_perf)
            i = min(int(round(elapsed_s / command_period_s)), len(goal_positions) - 1)
            if i == last_index:
                next_command_t += command_period_s
                continue

            goal_pos = goal_positions[i]
            success = self.write_goal_position(goal_pos)
            actual_position = self.read_present_position(max_retries=read_retries, sleep_s=0.0)
            motor_data['host_t_s'].append(time.perf_counter() - start_perf)
            motor_data['GoalPos'].append(goal_pos)
            motor_data['PresPos'].append(actual_position/self.angle_conversion if actual_position is not None else np.nan)
            motor_data['WriteOK'].append(success)
            last_index = i
            next_command_t = start_perf + (i + 1) * command_period_s

        return pd.DataFrame(motor_data)

    def move_to_position(self, goal_pos, timeout_s=2.0):
        self.write_goal_position(goal_pos)
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            actual_position = self.read_present_position(max_retries=1, sleep_s=0.0)
            if actual_position is None:
                time.sleep(0.01)
                continue
            if abs(goal_pos - actual_position) <= self.config.DXL_MOVING_STATUS_THRESHOLD:
                return True
            time.sleep(0.01)
        return False

    def close(self):
        try:
            self.port_num.closePort()
        except Exception:
            pass
