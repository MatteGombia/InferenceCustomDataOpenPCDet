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
        
        # print("Points with batch: ", np.shape(processed_points))

        return processed_points

    def alignRCSDistribution(self, rcs):
        rcs = ((rcs - RCS_MEAN) / RCS_STD) * VOD_RCS_STD + VOD_RCS_MEAN
        return rcs