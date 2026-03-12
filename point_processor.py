import numpy as np

use_SNR = False
ALIGN_RCS_DISTRIBUTION = True

class PointProcessor:
    def __init__(self, radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames):
        self.radar_offset_tx = radar_offset_tx
        self.radar_offset_ty = radar_offset_ty
        self.radar_offset_yaw = radar_offset_yaw

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_yaw = 0.0
        self.img = None

        self.n_frames = n_frames
        self.points_per_frame = []
        self.timestamp_last_frame = 0
        self.dt = 0
        self.multiframe_points = np.empty((0, 7), dtype=np.float32)  # [x, y, z, intensity, time, frame_id, velocity]

    def add_timestamp(self, timestamp):
        if self.timestamp_last_frame != 0:
            self.dt = (timestamp - self.timestamp_last_frame) * 1e-9  # Convert nanoseconds to seconds

        #print(f"New frame timestamp: {timestamp}, dt from last frame: {self.dt:.3f} seconds")
        
        self.timestamp_last_frame = timestamp

    def add_auxiliar_cloud(self, points, shift_x = 0.0, shift_y = 0.0, yaw = 0.0):
        print("**** DEBUG *********")
        print(f"Point 0 : {points[0]}, shift x, y, yaw {shift_x, shift_y, yaw}")
        yaw_in_radians = np.deg2rad(yaw)
        print(f"Yaw in radians {yaw_in_radians}")

        processed_points = self.processPointsSingleFrame(points, shift_x+self.radar_offset_tx, shift_y+self.radar_offset_ty, yaw_in_radians+self.radar_offset_yaw)
        print(f"Point 0 after processing: {processed_points[0]}")

        rotated_points = self.rotate_points(processed_points, shift_x, shift_y, yaw_in_radians)
        print(f"Point 0 after rotation: {rotated_points[0]}")

        #[x, y, z, snr, v_comp_x, v_comp_y, time]
        #processed_points = self.add_random_z(processed_points)
        #processed_points = self.snr_to_fake_rcs(processed_points)
        if use_SNR:
            rotated_points = self.convert_snr_to_rcs(rotated_points, C_ars430=68.0) # Example constant, should be calibrated
        else:
            rotated_points = self.convert_intensity_to_rcs(rotated_points)

        n_points_before = self.multiframe_points.shape[0]
        self.points_per_frame[-1] += rotated_points.shape[0]

        self.multiframe_points = np.vstack([self.multiframe_points, rotated_points])

        #print(f"Added {rotated_points.shape[0]} points from auxiliary cloud. Total points before: {n_points_before}, after: {self.multiframe_points.shape[0]}")

        #print(f"Tot points in vector: {sum(self.points_per_frame)}")
        #print(f"Current multiframe points shape: {self.multiframe_points.shape}")

        return 

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


    def calculate_compensated_velocity(self, points, shift_x, shift_y, shift_yaw):
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

        v_meas = points[:, 6] # The raw Doppler velocity from the radar
        

        v_x, v_y, omega_z = self.vel_x, self.vel_y, self.vel_yaw
        t_x, t_y, yaw = shift_x, shift_y, shift_yaw  
        
        # radar sensor's physical velocity
        v_sens_x = v_x - (omega_z * t_y)
        v_sens_y = v_y + (omega_z * t_x)
        
        # Rotate velocity into the radar's local coordinate frame
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        v_rad_x = v_sens_x * cos_y + v_sens_y * sin_y
        v_rad_y = -v_sens_x * sin_y + v_sens_y * cos_y
        
        # Distance from radar to point
        dist = np.sqrt(x**2 + y**2 + z**2)
        
        # Avoid division by zero for points exactly at (0,0,0)
        dist = np.clip(dist, a_min=1e-6, a_max=None)
        
        u_x = x / dist
        u_y = y / dist
        
        # Ego velocity
        v_ego_los = (v_rad_x * u_x) + (v_rad_y * u_y)
        
        # Compensate the raw measurement
        v_comp = v_meas + v_ego_los

        # print("EGO VELOCITY (X, Y, YAW): ", v_x, v_y, omega_z)
        # print("MEAN RADIAL VELOCITY BEFORE COMPENSATION: ", np.mean(v_meas))
        # print("MEAN EGO LOS VELOCITY: ", np.mean(v_ego_los))
        # print("MEAN RADIAL VELOCITY AFTER COMPENSATION: ", np.mean(v_comp))
        
        return v_comp
    
    def processPoints(self, points):
        processed_points = self.processPointsSingleFrame(points, self.radar_offset_tx, self.radar_offset_ty, self.radar_offset_yaw)

        #processed_points = self.add_random_z(processed_points)
        #processed_points = self.snr_to_fake_rcs(processed_points)

        processed_points = self.convert_intensity_to_rcs(processed_points)

        if len(self.points_per_frame) >= self.n_frames:
            self.multiframe_points = self.multiframe_points[self.points_per_frame[0]:, :]
            self.points_per_frame.pop(0)  # Remove the oldest frame

        

        self.points_per_frame.append(len(processed_points))

        self.multiframe_points = self.transposeFrame(self.multiframe_points)
        self.multiframe_points[:, 6] -= 1  # Decrease time for all points by 1
        self.multiframe_points = np.vstack([self.multiframe_points, processed_points])


        # print(f"Tot points in vector: {sum(self.points_per_frame)}")
        # print(f"Current multiframe points shape: {self.multiframe_points.shape}")
        # if len(self.points_per_frame) == 5:
        #     print(f"Sample of points frame 5: {self.multiframe_points[0:2, :]}")
        #     print(f"Sample of points frame 4: {self.multiframe_points[self.points_per_frame[0]:self.points_per_frame[0]+2, :]}")
        #     print(f"Sample of points frame 3: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]:self.points_per_frame[0]+self.points_per_frame[1]+2, :]}")
        #     print(f"Sample of points frame 2: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]:self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+2, :]}")
        #     print(f"Sample of points frame 1: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+self.points_per_frame[3]:self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+self.points_per_frame[3]+2, :]}")

        return self.multiframe_points
    
    def transposeFrame(self, points):
        # Apply the shift and rotation to the points
        cos_yaw = np.cos(self.vel_yaw * self.dt)
        sin_yaw = np.sin(self.vel_yaw * self.dt)

        x_shifted = points[:, 0] - self.vel_x * self.dt
        y_shifted = points[:, 1] - self.vel_y * self.dt

        x_rotated = x_shifted * cos_yaw + y_shifted * sin_yaw
        y_rotated = -x_shifted * sin_yaw + y_shifted * cos_yaw

        points[:, 0] = x_rotated
        points[:, 1] = y_rotated

        return points
    
    def processPointsSingleFrame(self, points, shift_x=0.0, shift_y=0.0, shift_yaw=0.0):
        v_comp = self.calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw)

        radial_ambiguous_velocity = np.expand_dims(points[:,6], axis=1)
        #print("Speed: ", np.shape(radial_ambiguous_velocity))
        v_comp=np.expand_dims(v_comp, axis=1)

        snr = np.expand_dims(points[:,3], axis=1)

        time_vector = np.zeros((points.shape[0], 1), dtype=points.dtype)
        processed_points = np.hstack([points[:, 0:3], snr, radial_ambiguous_velocity, v_comp, time_vector])
        
        # print("Points with batch: ", np.shape(processed_points))

        return processed_points

    

    def add_random_z(self,points):
    
        N = points.shape[0]
        
        # If the frame is completely empty, just return it
        if N == 0:
            return points

        # ==========================================
        # TRICK 1: THE Z-AXIS INFLATION
        # ==========================================
        
        points[:, 2] = np.random.uniform(0.3, 1.5, size=N)

        return points

    def snr_to_fake_rcs(self, points, snr_mean=None, snr_std=None):
        VOD_RCS_MEAN = 2.0
        VOD_RCS_STD = 12.0
        
        # Extract your raw SNR column
        snr = points[:, 3]
        
        # If not provided SNR stats, calculate them on the fly for this frame
        if snr_mean is None:
            snr_mean = np.mean(snr)
        if snr_std is None:
            snr_std = np.std(snr) + 1e-6 # Add tiny epsilon to prevent division by zero
            
        # Standardize SNR, then scale it to VoD's RCS distribution
        fake_rcs = ((snr - snr_mean) / snr_std) * VOD_RCS_STD + VOD_RCS_MEAN
        
        # Overwrite the SNR column with our fake RCS values
        points[:, 3] = fake_rcs
        
        return points

    def convert_intensity_to_rcs(self, points):
        MAX_RCS = 100
        MIN_RCS = -100

        rcs_norm = points[:, 3]  

        rcs = rcs_norm * (MAX_RCS - MIN_RCS) + MIN_RCS

        
        rcs_mean = np.mean(rcs)
        rcs_std = np.std(rcs)

        if ALIGN_RCS_DISTRIBUTION:
            
            VOD_RCS_MEAN = -12.43  
            VOD_RCS_STD = 13.27

            rcs = ((rcs - rcs_mean) / rcs_std) * VOD_RCS_STD + VOD_RCS_MEAN

            print("RCS stats - mean: {:.2f}, std: {:.2f}".format(np.mean(rcs), np.std(rcs)))
            # print("Sample RCS values: ", rcs[:10])

        points[:, 3] = rcs
        return points