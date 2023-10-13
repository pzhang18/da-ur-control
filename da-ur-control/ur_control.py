#!/usr/bin/env python
# Copyright (c) 2020-2022, Universal Robots A/S,
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Universal Robots A/S nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UNIVERSAL ROBOTS A/S BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import sys

sys.path.append("..")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import rtde.csv_writer as csv_writer
import rtde.csv_binary_writer as csv_binary_writer
import URBasic

import onnxruntime as ort
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", default="192.168.0.20", help="name of host to connect to (localhost)"
)
parser.add_argument("--port", type=int, default=30004, help="port number (30004)")
parser.add_argument(
    "--samples", type=int, default=0, help="number of samples to record"
)
parser.add_argument(
    "--frequency", type=int, default=10, help="the sampling frequency in Herz"
)
parser.add_argument(
    "--config",
    default="da_control_configuration.xml",
    help="data configuration file to use (da_control_configuration.xml)",
)
parser.add_argument(
    "--output",
    default="ur_control.csv",
    help="data output file to write to (ur_control.csv)",
)
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
parser.add_argument(
    "--buffered",
    help="Use buffered receive which doesn't skip data",
    action="store_true",
)
parser.add_argument(
    "--binary", help="save the data in binary format", action="store_true"
)
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)

conf = rtde_config.ConfigFile(args.config)
output_names, output_types = conf.get_recipe("out")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(args.host, args.port)
con.connect()

# get controller version
con.get_controller_version()

# setup recipes
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)
if not con.send_output_setup(output_names, output_types, frequency=args.frequency):
    logging.error("Unable to configure output")
    sys.exit()

# Setpoints to move the robot to
# setp1 = [0.58, 0.47, 0.41, 1.5, -2.9, 0.18]
# setp2 = [0.58, 0.5, 0.41, 1.5, -2.9, 0.18]

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

# The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_int_register_0 = 0

def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp

def to_rotation(vector):
    rx = vector[0]
    ry = vector[1]
    rz = vector[2]
    rotation_x_deg = np.arctan2(ry, rz)
    rotation_y_deg = np.arctan2(rx, rz)
    rotation_z_deg = np.arctan2(ry, rx)
    rotation = [rotation_x_deg,rotation_y_deg,rotation_z_deg]
    return rotation

def to_orientation(vector):
    # convert euler rotation angle (degrees) around xyz axis in unity 
    # to the tool orientation vector in cartesian coordinate
    roll = vector[0] # roll rotation_x_deg from unity
    yaw = vector[1] # yaw rotation_y_deg from unity
    pitch = vector[2] # pitch rotation_z_deg from unity
    yawMatrix = np.matrix([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
            ])
    pitchMatrix = np.matrix([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
            ])
    rollMatrix = np.matrix([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
            ])

    R = yawMatrix * pitchMatrix * rollMatrix
    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * math.sin(theta))
    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta
    return [rx, ry, rz]

# start data synchronization
if not con.send_start():
    logging.error("Unable to start synchronization")
    sys.exit()


# Load the ONNX model
model_path = "C:/Users/pengf/rl/da-ur-control/NN/SAC04_500k.onnx"
ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'],)

# open recording file
writeModes = "wb" if args.binary else "w"
with open(args.output, writeModes) as csvfile:
    writer = None

    if args.binary:
        writer = csv_binary_writer.CSVBinaryWriter(csvfile, output_names, output_types)
    else:
        writer = csv_writer.CSVWriter(csvfile, output_names, output_types)

    writer.writeheader()

    # Control loop
    delta = 0.001 # for debugging
    i = 1
    move_completed = True
    keep_running = True
    # Set Target
    obs_target_pos = [0.317, 0.403, 0.530,] # Manually measure vector3 
    obs_target_rotation = [-0.332, 1.670, 1.083] # vector3 
    # Debugging: obs_target_orientation = to_orientation(target_rotarion) # vector3

    while keep_running:

        if i % args.frequency == 0:
            if args.samples > 0:
                sys.stdout.write("\r")
                sys.stdout.write("{:.2%} done.".format(float(i) / float(args.samples)))
                sys.stdout.flush()
            else:
                sys.stdout.write("\r")
                sys.stdout.write("{:3d} samples.".format(i))
                sys.stdout.flush()
        if args.samples > 0 and i >= args.samples:
            keep_running = False
        try:
            if args.buffered:
                state = con.receive_buffered(args.binary)
            else:
                # Received state from Robot
                state = con.receive(args.binary)
            if state is not None:
                writer.writerow(state)
                i += 1

                # OUR CODE:
                # Get input from robot
                TCP_pose = state.__dict__["actual_TCP_pose"] # VECTOR6D
                TCP_force = state.__dict__["actual_TCP_force"] # VECTOR6D
                TCP_speed = state.__dict__["actual_TCP_speed"] # VECTOR6D
                print(f"TCPPose:{TCP_pose}")

                # ONNX inference and output actions
                # Prepare input data
                
                obs_position = [TCP_pose[0],TCP_pose[2],TCP_pose[1]] # vector3
                obs_rotation = [np.rad2deg(TCP_pose[3]),np.rad2deg(TCP_pose[5]),np.rad2deg(TCP_pose[4])] # vector3 from rad form RTDE to degree in unity
                obs_force = [TCP_force[0],TCP_force[2],TCP_force[1]] # vector3
                obs_torque = [TCP_force[3],TCP_force[5],TCP_force[4]] # vector3
                obs_velocity = [TCP_speed[0],TCP_speed[2],TCP_speed[1]] # vector3
                obs_ang_velocity = [TCP_speed[3],TCP_speed[5],TCP_speed[4]] # vector3
                # debugging: convert tool orientation to euler rotation. Not needed for control
                # obs_rotation = to_rotation(obs_orientation)
                # obs_target_rotation = to_rotation(obs_target_orientation)

                # concatenate all observation into an input tensor
                input_data = np.array((obs_target_pos + obs_target_rotation + obs_position + obs_rotation
                            + obs_force + obs_torque + obs_velocity + obs_ang_velocity),
                            dtype=np.float32)
                input_data = input_data.reshape(1, 24)
                ort_inputs = {ort_session.get_inputs()[0].name: input_data}
                ort_outputs = ort_session.run(None, ort_inputs) # gets actions from obs
                actions = ort_outputs[2][0]
                time_step = 1 / 100 # args.frequency
                act_velocity = [actions[0],actions[1],actions[2],] # keep unity XZY for calculate new position
                act_ang_velocity = [actions[3],actions[4],actions[5],] # keep unity XZY for calculate new rotation
                print("action speed:", act_velocity,act_velocity)

                # Send actions back to Robot
                if move_completed and state.output_int_register_0 == 1:
                    move_completed = False
                    # new_setp = setp1 if setp_to_list(setp) == setp2 else setp2
                    # delta = delta*(-1)
                    # new_setp= [sum(x) for x in zip (TCP_pose,[delta,0,0,0,0,0])]
                    new_position_unity = [sum(x) for x in zip(obs_position, [v*time_step for v in act_velocity])]  # 10HZ. in Unity XZY coordinate
                    new_position = [new_position_unity[0],new_position_unity[2],new_position_unity[1]] # UR XYZ coordinate
                    new_rotation_unity = [sum(x) for x in zip(obs_rotation, [w*time_step for w in act_ang_velocity])] # in Unity XZY coordinate
                    new_rotation = [new_rotation_unity[0],new_rotation_unity[2],new_rotation_unity[1]]
                    # new_orientation = to_orientation(new_rotation) # UR XYZ coordinate
                    new_setp = new_position + new_rotation
                    list_to_setp(setp, new_setp)
                    # print("New pose = " + str(new_setp))
                    # send new setpoint
                    con.send(setp)
                    watchdog.input_int_register_0 = 1
                elif not move_completed and state.output_int_register_0 == 0:
                    print("Move to confirmed pose = " + str(state.target_q))
                    move_completed = True
                    watchdog.input_int_register_0 = 0

                # Check if reached target

                # kick watchdog
                con.send(watchdog)

        except KeyboardInterrupt:
            keep_running = False
        except rtde.RTDEException:
            con.disconnect()
            sys.exit()


sys.stdout.write("\rComplete!            \n")

con.send_pause()
con.disconnect()
