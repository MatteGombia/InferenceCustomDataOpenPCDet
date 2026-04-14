import numpy as np
from base_point_processor import BasePointProcessor, RCS_MEAN, RCS_STD, use_SNR

VOD_RCS_MEAN = -12.43  
VOD_RCS_STD = 13.27

class PointProcessorVod(BasePointProcessor):
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

        return points
    
    def processPointsSingleFrame(self, points, timestamp_pc, shift_x=0.0, shift_y=0.0, shift_yaw=0.0):
        v_comp = self.calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw, timestamp_pc)

        radial_ambiguous_velocity = np.expand_dims(points[:,6], axis=1)
        #print("Speed: ", np.shape(radial_ambiguous_velocity))
        v_comp=np.expand_dims(v_comp, axis=1)

        if use_SNR:
            snr = np.expand_dims(points[:,4], axis=1)
        else:
            snr = self.convert_intensity_to_rcs(points[:,3])
            snr = np.expand_dims(snr, axis=1)

        time_vector = np.zeros((points.shape[0], 1), dtype=points.dtype)
        processed_points = np.hstack([points[:, 0:3], snr, radial_ambiguous_velocity, v_comp, time_vector])

        #processed_points = self.filterout_fixed_points(processed_points)
        processed_points = self.filter_invalid_points(processed_points)
        
        # print("Points with batch: ", np.shape(processed_points))

        return processed_points

    def alignRCSDistribution(self, rcs):
        rcs = ((rcs - RCS_MEAN) / RCS_STD) * VOD_RCS_STD + VOD_RCS_MEAN
        return rcs
    
    def filterout_fixed_points(self, points, fixed_threshold=0.6):
        # Assuming points is a numpy array of shape (N, 7) where the velocity component is at index 5
        v_comp = points[:, 5]
        mask = np.abs(v_comp) > fixed_threshold
        return points[mask]

    def updateTimestamp(self, timestamp):
        return timestamp - 1

    def filter_valid_speed_points(self, points):
        # Filter out points with unrealistic speeds (e.g., > 30 m/s)
        speed = np.abs(points[:, -2])
        valid_speed_points = points[speed < 30]
        return valid_speed_points