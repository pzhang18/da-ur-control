import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation

# Load the ONNX model
model_path = "C:/Users/pengf/rl/da-ur-control/NN/SAC04_500k.onnx"
ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'],)

# Prepare input data with 22 parameters 
# REPLACE WITH ACTUAL RTDE VALUES
obs_target_pos = [0.00424165790900588, 0.019999999552965164, 0.013266406953334808,] # Manually measure vector3
obs_target_orientation = [0.00020356, -0.00044415,  0.00088845] # vector3
obs_position = [0.0015319063095375896, 0.09820583462715149, -0.00041362192132510245, ] # vector3
obs_orientation = [0.00020356, -0.00044415,  0.00088845] # vector3
obs_force = [-1.3748521269008052e-05, -9.809981346130371, 4.723733582068235e-06, ] # vector3
obs_torque = [-0.0006215430330485106, 1.2438973726602853e-06, 0.4888324439525604,] # vector3
obs_velocity = [0.0, 0.0, 0.0,] # vector3
obs_ang_velocity = [-0.0011297821765765548, -0.0024679843336343765, 0.004935592878609896] # vector3

def to_rotation(vector):
    rx = vector[:, 0]
    ry = vector[:, 1]
    rz = vector[:, 2]
    rotation_x_deg = np.degrees(np.arctan2(ry, rz)).reshape(-1, 1)
    rotation_y_deg = np.degrees(np.arctan2(rx, rz)).reshape(-1, 1)
    rotation_z_deg = np.degrees(np.arctan2(ry, rx)).reshape(-1, 1)
    rotation = np.hstack((rotation_x_deg, rotation_y_deg, rotation_z_deg))
    return rotation

# convert tool orientation to euler rotation
obs_rotation = to_rotation(obs_orientation)
obs_target_rotation = to_rotation(obs_target_orientation)

# concatenate all observation into an input tensor
input_data = np.array((obs_target_pos + obs_target_rotation + obs_position + obs_rotation
               + obs_force + obs_torque + obs_velocity + obs_ang_velocity),
               dtype=np.float32)
input_data = input_data.reshape(1, 22)

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: input_data}
ort_outputs = ort_session.run(None, ort_inputs) # gets actions from obs

# Process the continuous actions
continuous_actions = ort_outputs[2][0]
print(continuous_actions)
# Assign output to action. Double check order of robot XYZ and unity XYZ
act_velocity_x = continuous_actions[0]
act_velocity_y = continuous_actions[1]
act_velocity_z = continuous_actions[2]
act_ang_velo_x = continuous_actions[3]
act_ang_velo_y = continuous_actions[4]
act_ang_velo_z = continuous_actions[5]

