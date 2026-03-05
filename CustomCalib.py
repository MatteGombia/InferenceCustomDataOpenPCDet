import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

radar_frame_id_default = "radar_fc"
camera_frame_id_default = "camera_f"

class CustomYAMLCalibration:
    def __init__(self, yaml_path, camera_intrinsic_matrix, radar_frame_id=radar_frame_id_default, camera_frame_id=camera_frame_id_default):
        self.K = np.array(camera_intrinsic_matrix, dtype=np.float32)
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        tfs = data['tree']['static_tf']
        
        # 1. Extract Transforms
        radar_tf = next(item for item in tfs if item['targetLink'] == radar_frame_id)
        camera_tf = next(item for item in tfs if item['targetLink'] == camera_frame_id)
        
        # 2. Convert [x, y, z, roll, pitch, yaw] to 4x4 Transformation Matrices
        T_axle_to_radar = self._tf_to_matrix(radar_tf['tf'])
        T_axle_to_camera = self._tf_to_matrix(camera_tf['tf'])
        
        # 3. Calculate Radar -> Camera (ROS Frame)
        T_camera_to_axle = np.linalg.inv(T_axle_to_camera)
        self.T_radar_to_camera_ros = np.dot(T_camera_to_axle, T_axle_to_radar)

        # 4. THE MAGIC FIX: ROS Frame -> Optical Frame
        # Maps: X(forward)->Z(depth), Y(left)-> -X(right), Z(up)-> -Y(down)
        self.T_ros_to_optical = np.array([
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0,  0.0],
            [ 1.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0]
        ])

    def _tf_to_matrix(self, tf_array):
        x, y, z, roll, pitch, yaw = tf_array
        
        # Change degrees=False to degrees=True!
        rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [x, y, z]
        return T

    def lidar_to_img(self, pts_3d):
        """ Projects (N, 3) 3D points to 2D pixels """
        # 1. To Homogeneous (N, 4)
        pts_3d_homog = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
        
        # 2. Radar -> Camera Link (ROS Coordinates)
        pts_cam_ros = np.dot(pts_3d_homog, self.T_radar_to_camera_ros.T)
        
        # 3. Camera Link -> Optical Frame (Z is forward!)
        pts_cam_opt = np.dot(pts_cam_ros, self.T_ros_to_optical.T)
        pts_cam_opt_3d = pts_cam_opt[:, :3]
        
        # 4. Optical Frame -> 2D Image Pixels (Using Intrinsics)
        pts_img_homog = np.dot(pts_cam_opt_3d, self.K.T)
        
        depths = pts_img_homog[:, 2]
        depths_safe = np.copy(depths)
        depths_safe[depths_safe == 0] = 1e-5
        
        pts_2d = np.zeros((pts_3d.shape[0], 2))
        pts_2d[:, 0] = pts_img_homog[:, 0] / depths_safe # u (x-pixel)
        pts_2d[:, 1] = pts_img_homog[:, 1] / depths_safe # v (y-pixel)
        
        return pts_2d, depths