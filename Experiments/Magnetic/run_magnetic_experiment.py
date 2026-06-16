import os, sys, time, csv, subprocess, tempfile, pathlib, threading
from string import Template
from dataclasses import dataclass
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import dynamixel_sdk as dynamixel                    # Uses DYNAMIXEL SDK library

from sensing import ArduinoSensing
from control import MotorControl

@dataclass
class SensingConfig:
    channels: tuple = (("A0", "pickup1"), ("A1", "pickup2"), ("A2", "pickup3"), ("A3", "pickup4"))
    VCC: float = 5.0 # volts
    DURATION_MS: int = 60_000 # milliseconds
    CAL_TIME_MIN_MS: int = 5_000 # milliseconds
    CAL_SAMPLES_PER_CH: int = 2000 # samples
    SAMPLE_RATE_HZ: int = 500 # samples per second
    OVERSAMPLE: int = 16 # oversampling factor
    USE_SMOOTH: bool = True # whether to use exponential moving average (EMA) smoothing
    ALPHA: float = 0.2 # smoothing factor, increasing will reduce smoothing (only used if USE_SMOOTH is True)
    BAUD: int = 250000 # baud rate
    PORT: str = "COM8" # serial port
    FQBN: str = "arduino:avr:uno" # fully qualified board name
    ARDUINO_CLI: str = "C:\\arduino-cli_1.5.0_Windows_64bit\\arduino-cli.exe" # if not on PATH, put full path to arduino-cli here
    BASE_DIR: str = "./Experiments/Magnetic/" # name of the Experiment Directory
    EXP_NAME: str = "magnetic_four_pickups_2x2" # name of the Arduino sketch (without .ino) and csv/png output stem
    OUTPUT_STEM: str = os.path.join(BASE_DIR, EXP_NAME) # CSV/PNG base path
    LIVE_PLOT: bool = True # whether to show live plot during acquisition (requires matplotlib)
    X_SECONDS: int = 60 # fixed x-axis limits in seconds
    Y_LIMS: tuple = (-0.1, 0.1) # fixed y-axis limits in volts (set to None for auto-scaling)

    def __init__(self, duration_ms=None, **kwargs):
        if duration_ms is not None:
            self.DURATION_MS = duration_ms
        self.channels = kwargs.get("channels", self.channels)
        self.EXP_NAME = kwargs.get("EXP_NAME", self.EXP_NAME)

@dataclass
class MotorControlConfig:
    # Control table address
    ADDR_MX_TORQUE_ENABLE       = 24                            # Control table address is different in Dynamixel model
    ADDR_MX_GOAL_POSITION       = 30
    ADDR_MX_PRESENT_POSITION    = 36
    ADDR_MX_MOVING_SPEED        = 32

    # Protocol version
    PROTOCOL_VERSION            = 1                             # See which protocol version is used in the Dynamixel

    # Default setting
    DXL_ID                      = 1                             # Dynamixel ID: 1
    BAUDRATE                    = 1000000
    DEVICENAME                  = "COM9"                        # Check which port is being used on your controller
                                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0"

    TORQUE_ENABLE               = 1                             # Value for enabling the torque
    TORQUE_DISABLE              = 0                             # Value for disabling the torque
    DXL_MINIMUM_POSITION_VALUE  = 0                             # Dynamixel will rotate between this value
    DXL_MAXIMUM_POSITION_VALUE  = 1023                          # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
    DXL_MOVING_STATUS_THRESHOLD = 10                            # Dynamixel moving status threshold

    ESC_ASCII_VALUE             = 0x1b

    COMM_SUCCESS                = 0                             # Communication Success result value
    COMM_TX_FAIL                = -1001                         # Communication Tx Failed

def run_synchronized_experiment(sensor, motor, motor_speed_rpm, goal_positions, duration_s, rest_position_deg=None, arduino_start_delay_s=0.0):
    ser = None
    stop_event = threading.Event()
    sensor_result = {}
    sensor_error = {}
    motor_df = pd.DataFrame()

    try:
        ser = sensor.calibrate_sensor()
        motor.connect_and_enable_motor()
        motor.set_moving_speed_rpm(motor_speed_rpm) # Set a moderate moving speed to ensure we can track the position accurately

        ans = input("Sensor calibrated and motor enabled. Start synchronized run? [y/N]: ").strip().lower()
        if not ans.startswith("y"):
            ser.close()
            print("Ok--skipping synchronized run.")
            return pd.DataFrame(), pd.DataFrame()

        start_event = threading.Event()
        start_holder = {}

        def sensor_worker():
            try:
                start_event.wait()
                # Keep plotting off during synchronized acquisition to avoid UI latency.
                sensor_result["data"] = sensor.acquire_started_run(
                    ser, start_holder["start_perf"], live_plot=False, stop_event=stop_event
                )
            except Exception as exc:
                sensor_error["error"] = exc
                stop_event.set()

        sensor_thread = threading.Thread(target=sensor_worker, name="sensor_logger")
        sensor_thread.start()

        start_command_perf = sensor.start_sensing_run(ser)
        start_perf = start_command_perf + arduino_start_delay_s
        start_holder["start_perf"] = start_perf
        start_event.set()

        try:
            motor_df = motor.run_motor_profile_and_log(
                goal_positions, start_perf=start_perf, duration_s=duration_s, stop_event=stop_event
            )
        except Exception:
            stop_event.set()
            raise
        finally:
            if rest_position_deg is not None:
                rest_pos = int(rest_position_deg * motor.angle_conversion)
                motor.move_to_position(rest_pos, timeout_s=2.0)

            try:
                motor.disable_torque()
            except Exception as exc:
                print(f"Warning: could not disable motor torque cleanly: {exc}")

        sensor_thread.join(timeout=5.0)
        if sensor_thread.is_alive():
            try:
                ser.write(b"X")
            except Exception:
                pass
            sensor_thread.join(timeout=2.0)

        if "error" in sensor_error:
            raise sensor_error["error"]

        outstem = pathlib.Path(sensor.config.OUTPUT_STEM)
        motor_csv_path = f"{outstem}_motor.csv"
        motor_df.to_csv(motor_csv_path, index=False)
        print(f"Saved motor CSV: {motor_csv_path}")

        sensor_df = sensor_result.get("data", pd.DataFrame())
        if not sensor_df.empty and not motor_df.empty:
            combined_csv_path = f"{outstem}_combined.csv"
            motor_for_merge = motor_df.rename(columns={"host_t_s": "motor_t_s"})
            combined_df = pd.merge_asof(
                sensor_df.sort_values("sensor_t_s"),
                motor_for_merge.sort_values("motor_t_s"),
                left_on="sensor_t_s",
                right_on="motor_t_s",
                direction="nearest",
                tolerance=0.02,
            )
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"Saved combined CSV: {combined_csv_path}")

        return sensor_df, motor_df, combined_df

    except KeyboardInterrupt:
        stop_event.set()
        if ser is not None:
            try:
                ser.write(b"X")
            except Exception:
                pass
        raise
    finally:
        motor.close()          
        
if __name__ == "__main__":

    channels = (("A0", "pickup1"), ("A1", "pickup2"), ("A2", "pickup3"), ("A3", "pickup4"))
    
    duration = 60 # seconds
    mean_position_deg = 205
    rest_position_deg = mean_position_deg
    amplitude_deg = 30
    seed = 1234
    eval_freq = 100
    motor_speed_rpm = 50
    sample_freq = np.rint(len(channels) * 10 / duration).astype(int) # at least 10 samples per channel per second

    filename = f"magnetic_amp{amplitude_deg}_speed{motor_speed_rpm}rpm_sampfreq{sample_freq}Hz"

    sensor_config = SensingConfig(duration_ms=duration*1000, channels=channels, EXP_NAME=filename)
    motor_config = MotorControlConfig()

    sensor = ArduinoSensing(sensor_config)
    sketch_dir, ino_path = sensor.gen_sketch_dir()
    code = sensor.build_arduino_code()
    with open(ino_path, "w", newline="\n") as f:
        f.write(code)
    sensor.compile_upload(sketch_dir)

    motor = MotorControl(motor_config)
    motor_goal_positions = motor.generate_goal_positions(amplitude_deg, mean_position_deg, duration, sample_freq, seed, eval_freq=eval_freq)

    ans = input("Upload OK. Calibrate sensor now? [y/N]: ").strip().lower()
    if ans.startswith("y"):
        sensor_df, motor_df, combined_df = run_synchronized_experiment(
            sensor,
            motor,
            motor_speed_rpm,
            motor_goal_positions,
            duration_s=duration,
            rest_position_deg=rest_position_deg,
        )

        fig, axs = plt.subplots(5, 1, figsize=(20, 25), sharex=True)
        axs[0].plot(combined_df["sensor_t_s"], combined_df["PresPos"], label="Motor Position (deg)")
        axs[0].set_ylabel("Angle (deg)")
        axs[0].set_title("Motor Position vs. Time")

        axs[1].plot(combined_df["sensor_t_s"], combined_df["pickup1"], label="Pickup 1")
        axs[1].set_ylabel("Voltage (V)")
        axs[1].set_title("Pickup 1 vs. Time")

        axs[2].plot(combined_df["sensor_t_s"], combined_df["pickup2"], label="Pickup 2")
        axs[2].set_ylabel("Voltage (V)")
        axs[2].set_title("Pickup 2 vs. Time")

        axs[3].plot(combined_df["sensor_t_s"], combined_df["pickup3"], label="Pickup 3")
        axs[3].set_ylabel("Voltage (V)")
        axs[3].set_title("Pickup 3 vs. Time")

        axs[4].plot(combined_df["sensor_t_s"], combined_df["pickup4"], label="Pickup 4")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Voltage (V)")
        axs[4].set_title("Pickup 4 vs. Time")

        plt.savefig(f"{sensor_config.OUTPUT_STEM}_combined_plot.png")
        plt.show()
    else:
        print("Ok—skipping run.")

    
