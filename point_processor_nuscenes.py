import numpy as np
from base_point_processor import BasePointProcessor, RCS_MEAN, RCS_STD, use_SNR

NUSCENE_RCS_MEAN = 6.90
NUSCENE_RCS_STD = 7.60

class PointProcessorNuscenes(BasePointProcessor):
    def __init__(self, radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames):
        super().__init__(radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames)

    

    def rotate_points(self, points, shift_x, shift_y, yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 1. Rotate FIRST
        x_rotated = points[:, 0] * cos_yaw - points[:, 1] * sin_yaw
        y_rotated = points[:, 0] * sin_yaw + points[:, 1] * cos_yaw

        # 2. Add translation SECOND
        points[:, 0] = x_rotated + shift_x
        points[:, 1] = y_rotated + shift_y

        # 3. Rotate Velocities 
        # Velocities only rotate
        vx = points[:, 4].copy()
        vy = points[:, 5].copy()
        
        points[:, 4] = vx * cos_yaw - vy * sin_yaw
        points[:, 5] = vx * sin_yaw + vy * cos_yaw

        return points


    def calculate_compensated_velocity(self, points, shift_x, shift_y, shift_yaw, timestamp_pc):
        """
        Calculates the absolute compensated radial velocity for radar points.
        
        Args:
            points: (N, 7) numpy array where columns are [x, y, z, intensity, RCS, max_v_r, v_r, v_r_comp, time]
        
        Returns:
            v_comp: (N,) numpy array of compensated velocities
        """
        v_comp = self.super().calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw, timestamp_pc)
        v_comp_x = v_comp * self.u_x
        v_comp_y = v_comp * self.u_y

        v_comp_list = np.column_stack([v_comp_x, v_comp_y])


        return v_comp_list
    
    
    def processPointsSingleFrame(self, points, timestamp_pc, shift_x=0.0, shift_y=0.0, shift_yaw=0.0):
        v_comp = self.calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw, timestamp_pc)

        #print("Speed: ", np.shape(radial_ambiguous_velocity))
        #v_comp=np.expand_dims(v_comp, axis=1)
        if use_SNR:
            snr = np.expand_dims(points[:,4], axis=1)
        else: #Intensity
            snr = self.convert_intensity_to_rcs(points[:,3])
            snr = np.expand_dims(snr, axis=1)

        time_vector = np.zeros((points.shape[0], 1), dtype=points.dtype)
        processed_points = np.hstack([points[:, 0:3], snr, v_comp, time_vector])
        
        # [x, y, z, snr, v_comp_x, v_comp_y, time]
        
        print("Processed points shape: ", np.shape(processed_points))

        return processed_points
    
    def alignRCSDistribution(self, rcs):
        rcs = ((rcs - RCS_MEAN) / RCS_STD) * NUSCENE_RCS_STD + NUSCENE_RCS_MEAN
        return rcs

    