from nuscenes.nuscenes import *
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

def calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw, radar_offset_tx=0.0, radar_offset_ty=0.0, radar_offset_yaw=0.0):
    """
    Calculates the absolute compensated radial velocity for radar points.
    
    Args:
        points: (N, 7) numpy array where columns are [x, y, z, intensity, RCS, max_v_r, v_r, v_r_comp, time]
    
    Returns:
        v_comp: (N,) numpy array of compensated velocities
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]


    v_meas_x = points[:, 3] # The raw Doppler velocity from the radar
    v_meas_y = points[:, 4] # The raw Doppler velocity from the radar

    v_x, v_y, omega_z = shift_x, shift_y, shift_yaw
    t_x, t_y, yaw = radar_offset_tx, radar_offset_ty, radar_offset_yaw  
    
    # radar sensor's physical velocity
    v_sens_x = v_x - (omega_z * t_y)
    v_sens_y = v_y + (omega_z * t_x)
    print(f"DEBUG: X,Y,Z {x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f}")
    print(f"DEBUG: v_x: {v_x:.2f} m/s, v_y: {v_y:.2f} m/s, omega_z: {omega_z:.4f} rad/s")
    print(f"DEBUG: v_sens_x: {v_sens_x:.2f} m/s, v_sens_y: {v_sens_y:.2f} m/s")
    
    # Rotate velocity into the radar's local coordinate frame
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    v_rad_x = v_sens_x * cos_y + v_sens_y * sin_y
    v_rad_y = -v_sens_x * sin_y + v_sens_y * cos_y
    print(f"DEBUG: v_rad_x: {v_rad_x:.2f} m/s, v_rad_y: {v_rad_y:.2f} m/s")
    
    # Distance from radar to point
    dist = np.sqrt(x**2 + y**2 + z**2)
    print(f"DEBUG: dist: {dist[0]:.2f} m")
    
    # Avoid division by zero for points exactly at (0,0,0)
    dist = np.clip(dist, a_min=1e-6, a_max=None)
    
    u_x = x / dist
    u_y = y / dist

    print(f"DEBUG: u_x: {u_x[0]:.4f}, u_y: {u_y[0]:.4f}")
    
    # Ego velocity
    v_ego_los = (v_rad_x * u_x) + (v_rad_y * u_y)
    print(f"DEBUG: v_ego_los: {v_ego_los} m/s")

    v_rad = v_meas_x * u_x + v_meas_y * u_y
    v_comp_rad = v_rad + v_ego_los
    v_comp_x = v_comp_rad * u_x
    v_comp_y = v_comp_rad * u_y

    print(f"DEBUG: v_meas_x: {v_meas_x[0]:.2f} m/s, v_meas_y: {v_meas_y[0]:.2f} m/s")
    print(f"DEBUG: v_comp_x: {v_comp_x[0]:.2f} m/s, v_comp_y: {v_comp_y[0]:.2f} m/s")

    
    # Compensate the raw measurement
    v_comp = np.column_stack([v_comp_x, v_comp_y])
    
    return v_comp




nusc = NuScenes(version='v1.0-trainval', dataroot='/media/franco/hdd/dataset/nuscenes', verbose=True)
my_scene = nusc.scene[0]
first_sample = nusc.get('sample', my_scene['first_sample_token'])
my_sample = nusc.get('sample', first_sample['next']) 

sensor = 'RADAR_FRONT'
current_sd = nusc.get('sample_data', my_sample['data'][sensor])
prev_sd = nusc.get('sample_data', current_sd['prev'])

curr_pose = nusc.get('ego_pose', current_sd['ego_pose_token'])
prev_pose = nusc.get('ego_pose', prev_sd['ego_pose_token'])

# --- EGO VELOCITY EXTRACTION ---
dt = (curr_pose['timestamp'] - prev_pose['timestamp']) / 1e6

# Global motion vector
v_global = (np.array(curr_pose['translation']) - np.array(prev_pose['translation'])) / dt

#  Global -> local
v_ego = Quaternion(curr_pose['rotation']).inverse.rotate(v_global)
shift_x, shift_y = v_ego[0], v_ego[1]

yaw_curr = quaternion_yaw(Quaternion(curr_pose['rotation']))
yaw_prev = quaternion_yaw(Quaternion(prev_pose['rotation']))
dyaw = yaw_curr - yaw_prev
if dyaw > np.pi: dyaw -= 2 * np.pi
if dyaw < -np.pi: dyaw += 2 * np.pi
shift_yaw = dyaw / dt

cs_record = nusc.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
radar_offset_tx, radar_offset_ty, _ = cs_record['translation']

radar_offset_yaw = quaternion_yaw(Quaternion(cs_record['rotation']))

print(f"DEBUG INPUTS: Ego Forward Speed: {shift_x:.2f} m/s | Radar Yaw Offset: {radar_offset_yaw:.2f} rad")

# --- LOAD AND TEST ---
radar_pc = RadarPointCloud.from_file(nusc.get_sample_data_path(current_sd['token']))
print(f"Loaded radar point cloud: {radar_pc.points.T[:, 15]}")
points = np.hstack([radar_pc.points.T[:, :3], radar_pc.points.T[:, 6:10]])
print(f"Constructed points array (x, y, z, v_x, v_y, v_comp_x, v_comp_y): {points[0:3]}")

rcs = radar_pc.points.T[:, 5]
time
print("Syntetic RCS stats - mean: {:.2f}, std: {:.2f}".format(np.mean(rcs), np.std(rcs)))
print("Sample synthetic RCS values: ", rcs[:10])

v_comp_pred = calculate_compensated_velocity(
    points, 
    shift_x, shift_y, shift_yaw, 
    radar_offset_tx, radar_offset_ty, 0.0
)

mean_x_error = np.abs(points[:, 5] - v_comp_pred[:, 0]).mean()
mean_y_error = np.abs(points[:, 6] - v_comp_pred[:, 1]).mean()

print(f"\n--- INPUTS FED TO YOUR FUNCTION ---")
print(f"Ego Velocity (X, Y): {shift_x:.2f} m/s, {shift_y:.2f} m/s")
print(f"Ego Yaw Rate: {shift_yaw:.4f} rad/s")
print(f"Radar Offsets (X, Y, Yaw): {radar_offset_tx:.2f}m, {radar_offset_ty:.2f}m, {radar_offset_yaw:.2f}rad")

print(f"\n--- ERROR RESULTS ---")
print(f"MEAN ERROR IN X: {mean_x_error:.6f}")
print(f"MEAN ERROR IN Y: {mean_y_error:.6f}")
print(f"MEAN RADIAL VELOCITY BEFORE COMPENSATION ON X: {np.mean(points[:, 3]):.2f} m/s")
print(f"MEAN RADIAL VELOCITY BEFORE COMPENSATION ON Y: {np.mean(points[:, 4]):.2f} m/s")
print(f"MEAN RADIAL VELOCITY AFTER COMPENSATION GT: {np.mean(points[:, 5]):.2f} m/s (X), {np.mean(points[:, 6]):.2f} m/s (Y)")
print(f"MEAN RADIAL VELOCITY AFTER COMPENSATION: {np.mean(v_comp_pred[:, 0]):.2f} m/s (X), {np.mean(v_comp_pred[:, 1]):.2f} m/s (Y)")

print(f"Comp X vector: {v_comp_pred[:, 0]}")
print(f"Difference on X: {points[:, 5] - v_comp_pred[:, 0]} m/s")

#x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms