import numpy as np

use_SNR = False

class PointProcessorNuscenes:
    def __init__(self, radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames):
        self.radar_offset_tx = radar_offset_tx
        self.radar_offset_ty = radar_offset_ty
        self.radar_offset_yaw = radar_offset_yaw

        self.shift_x = 0.0
        self.shift_y = 0.0
        self.shift_yaw = 0.0
        self.img = None

        self.n_frames = n_frames
        self.points_per_frame = []
        self.timestamp_last_frame = 0
        self.dt = 0
        self.multiframe_points = np.empty((0, 7), dtype=np.float32) 

    def add_timestamp(self, timestamp):
        if self.timestamp_last_frame != 0:
            self.dt = (timestamp - self.timestamp_last_frame) * 1e-9  # Convert nanoseconds to seconds

        print(f"New frame timestamp: {timestamp}, dt from last frame: {self.dt:.3f} seconds")
        
        self.timestamp_last_frame = timestamp


    def calculate_compensated_velocity(self, points):
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
        

        v_x, v_y, omega_z = self.shift_x, self.shift_y, self.shift_yaw
        t_x, t_y, yaw = self.radar_offset_tx, self.radar_offset_ty, self.radar_offset_yaw  
        
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

        v_comp_rad = v_meas + v_ego_los

        v_comp_x = v_comp_rad * u_x
        v_comp_y = v_comp_rad * u_y

        # Compensate the raw measurement
        v_comp = np.column_stack([v_comp_x, v_comp_y])

        print("EGO VELOCITY (X, Y, YAW): ", v_x, v_y, omega_z)
        print("MEAN RADIAL VELOCITY BEFORE COMPENSATION: ", np.mean(v_meas))
        print("MEAN EGO LOS VELOCITY: ", np.mean(v_ego_los))
        print("MEAN RADIAL VELOCITY AFTER COMPENSATION: ", np.mean(v_comp[:, 0]))
        print("MEAN RADIAL VELOCITY AFTER COMPENSATION: ", np.mean(v_comp[:, 1]))
        print("SHAPE COMPENSATED VELOCITY: ", v_comp.shape)
        
        return v_comp
    
    def processPoints(self, points):
        # #DEBUG to find costant C
        # processed_points = self.processPointsSingleFrame(points)
        # if sum(self.points_per_frame) > 50000:
        #     self.calib_constant(self.multiframe_points)
        # else:
        #     self.points_per_frame.append(len(processed_points))
        #     self.multiframe_points = np.vstack([self.multiframe_points, processed_points])
        # return self.multiframe_points
        # #
        processed_points = self.processPointsSingleFrame(points)

        #processed_points = self.add_random_z(processed_points)
        #processed_points = self.snr_to_fake_rcs(processed_points)
        if use_SNR:
            processed_points = self.convert_snr_to_rcs(processed_points, C_ars430=68.0) # Example constant, should be calibrated
        else:
            processed_points = self.convert_intensity_to_rcs(processed_points)

        if len(self.points_per_frame) >= self.n_frames:
            self.multiframe_points = self.multiframe_points[self.points_per_frame[0]:, :]
            self.points_per_frame.pop(0)  # Remove the oldest frame

        

        self.points_per_frame.append(len(processed_points))

        self.multiframe_points = self.transposeFrame(self.multiframe_points)
        self.multiframe_points[:, 6] -= self.dt  # Decrease time for all points by the difference in timestamp
        self.multiframe_points = np.vstack([self.multiframe_points, processed_points])


        print(f"Tot points in vector: {sum(self.points_per_frame)}")
        print(f"Current multiframe points shape: {self.multiframe_points.shape}")
        # if len(self.points_per_frame) == 5:
        #     print(f"Sample of points frame 5: {self.multiframe_points[0:2, :]}")
        #     print(f"Sample of points frame 4: {self.multiframe_points[self.points_per_frame[0]:self.points_per_frame[0]+2, :]}")
        #     print(f"Sample of points frame 3: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]:self.points_per_frame[0]+self.points_per_frame[1]+2, :]}")
        #     print(f"Sample of points frame 2: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]:self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+2, :]}")
        #     print(f"Sample of points frame 1: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+self.points_per_frame[3]:self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+self.points_per_frame[3]+2, :]}")

        return self.multiframe_points
    
    def transposeFrame(self, points):
        # Apply the shift and rotation to the points
        cos_yaw = np.cos(self.shift_yaw * self.dt)
        sin_yaw = np.sin(self.shift_yaw * self.dt)

        x_shifted = points[:, 0] - self.shift_x * self.dt
        y_shifted = points[:, 1] - self.shift_y * self.dt

        x_rotated = x_shifted * cos_yaw + y_shifted * sin_yaw
        y_rotated = -x_shifted * sin_yaw + y_shifted * cos_yaw

        points[:, 0] = x_rotated
        points[:, 1] = y_rotated

        return points
    
    def processPointsSingleFrame(self, points):
        v_comp = self.calculate_compensated_velocity(points)
        
        #print("Speed: ", np.shape(radial_ambiguous_velocity))
        #v_comp=np.expand_dims(v_comp, axis=1)
        if use_SNR:
            snr = np.expand_dims(points[:,4], axis=1)
        else: #Intensity
            snr = np.expand_dims(points[:,3], axis=1)

        time_vector = np.zeros((points.shape[0], 1), dtype=points.dtype)
        processed_points = np.hstack([points[:, 0:3], snr, v_comp, time_vector])
        
        # [x, y, z, snr, v_comp_x, v_comp_y, time]
        
        print("Processed points shape: ", np.shape(processed_points))

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

    # def snr_to_fake_rcs(self, points, snr_mean=None, snr_std=None):
    #     VOD_RCS_MEAN = 2.0
    #     VOD_RCS_STD = 12.0
        
    #     # Extract your raw SNR column
    #     snr = points[:, 3]
        
    #     # If not provided SNR stats, calculate them on the fly for this frame
    #     if snr_mean is None:
    #         snr_mean = np.mean(snr)
    #     if snr_std is None:
    #         snr_std = np.std(snr) + 1e-6 # Add tiny epsilon to prevent division by zero
            
    #     # Standardize SNR, then scale it to VoD's RCS distribution
    #     fake_rcs = ((snr - snr_mean) / snr_std) * VOD_RCS_STD + VOD_RCS_MEAN
        
    #     # Overwrite the SNR column with our fake RCS values
    #     points[:, 3] = fake_rcs
        
    #     return points

    def convert_intensity_to_rcs(self, points):
        MAX_RCS = 100
        MIN_RCS = -100

        rcs_norm = points[:, 3]  

        rcs = rcs_norm * (MAX_RCS - MIN_RCS) + MIN_RCS

        points[:, 3] = rcs

        print("RCS stats - mean: {:.2f}, std: {:.2f}".format(np.mean(rcs), np.std(rcs)))
        print("Sample RCS values: ", rcs[:10])

        return points

        

    def convert_snr_to_rcs(self,points, C_ars430):
        """
        points: (N, 7) array [x, y, z, snr, vx, vy, time]
        """
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        snr = points[:, 3]
        
        # Calculate radial distance
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.clip(r, a_min=1.0, a_max=None) # Prevent log(0)
        
        # Invert the radar equation
        synthetic_rcs = snr + 40 * np.log10(r) - C_ars430
        
        SYNTH_MEAN = -4.66
        SYNTH_STD = 24.39
        
        # nuScenes' exact stats
        NUSC_MEAN = 6.90
        NUSC_STD = 7.60
        
        # 2. Statistics: Z-Score matching
        # Standardize to mean=0, std=1
        z_score = (synthetic_rcs - SYNTH_MEAN) / SYNTH_STD 
        
        # Project into nuScenes distribution
        rcs_aligned = (z_score * NUSC_STD) + NUSC_MEAN
        
        # 3. Hardware Emulation: Quantize to nearest 0.5 (just like ARS408)
        rcs_final = np.round(rcs_aligned * 2.0) / 2.0

        # Replace SNR column with Synthetic RCS
        points[:, 3] = rcs_final

        print("Syntetic RCS stats - mean: {:.2f}, std: {:.2f}".format(np.mean(synthetic_rcs), np.std(synthetic_rcs)))
        print("Sample synthetic RCS values: ", synthetic_rcs[:10])
        
        return points
    
    def calib_constant(self, points):
        # --- 2. Calculate Uncalibrated RCS ---
        # Get the distance to every point

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        snr = points[:, 3]

        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.clip(r, a_min=1.0, a_max=None) # Prevent log10(0) crash

        # Apply the physics formula WITHOUT the constant
        uncalibrated_rcs = snr + 40 * np.log10(r)

        # --- 3. Align the Medians ---
        # The known median RCS of the entire nuScenes dataset is roughly 3.0 dBsm
        NUSCENES_MEDIAN_RCS = 3.0

        custom_median = np.median(uncalibrated_rcs)

        # Calculate your hardware constant
        C_ars430 = custom_median - NUSCENES_MEDIAN_RCS

        print(f"Your custom data's uncalibrated median is: {custom_median:.2f}")
        print(f"-> Your ARS430 Hardware Constant (C) is: {C_ars430:.2f}")