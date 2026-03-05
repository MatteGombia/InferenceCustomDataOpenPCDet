import sys
import numpy as np
from mcap_protobuf.decoder import DecoderFactory
from mcap.reader import make_reader
from cloud import protoCloudToNumpy
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets.dataset import DatasetTemplate
import logging
import torch
from visualization_2D_custom import Visualization2D
from CustomCalib import CustomYAMLCalibration
from vis_tools import saveODImgs
from vod.visualization.settings import label_color_palette_2d

import cv2

# Format: 
# [ f_x,   0.0,  c_x ]
# [ 0.0,   f_y,  c_y ]
# [ 0.0,   0.0,  1.0 ]

# For Marzaglia, the camera intrinsics are actually not precise
# my_camera_intrinsics = [
#     [1200.0,    0.0,  960.0],
#     [   0.0, 1200.0,  540.0],
#     [   0.0,    0.0,    1.0]
# ]

# For 4porte, the camera intrinsics are actually not precise
my_camera_intrinsics = [
    [800.0,   0.0, 400.0],
    [  0.0, 800.0, 300.0],
    [  0.0,   0.0,   1.0]
]

# Marzaglia
# radar_frame_id = "radar_fc"
# camera_frame_id = "camera_fcl"

# 4porte
radar_frame_id = "radar_fc"
camera_frame_id = "cam_f"

# 4porte
RADAR_OFFSET_TX = 3.5  # meters forward from the vehicle's center
RADAR_OFFSET_TY = -0.5  # meters to the left of the vehicle
RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

OUTPUT_DIR_IMGS_2D = "./results/2D/images"
IMG_PATH_BEV = "/media/franco/hdd/matteogombia/OpenPCDet/tools/results/BEV/images"
#YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte_marzaglia.yaml"
YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte.yaml"
CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar.yaml"
CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar/default/ckpt/checkpoint_epoch_140.pth" 

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, 
            root_path=root_path, logger=logger
        )
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return {}

class DataProcessor:
    def __init__(self):
        self.img = None
        self.cloud = None
        self.counter = 0

        self.shift_x = 0.0
        self.shift_y = 0.0
        self.shift_yaw = 0.0

        calib = CustomYAMLCalibration(
            yaml_path=YAML_PATH, 
            camera_intrinsic_matrix=my_camera_intrinsics,
            radar_frame_id=radar_frame_id,
            camera_frame_id=camera_frame_id
        )

        self.vis = Visualization2D(
            calib=calib,
            output_dir=OUTPUT_DIR_IMGS_2D,
        )
        
        # Load config
        cfg_from_yaml_file(CFG_FILE, cfg)

        #Dummy dataset and dataloader
        self.dummy_dataset = CustomDataset(
            dataset_cfg=cfg.DATA_CONFIG, 
            class_names=cfg.CLASS_NAMES, 
            training=False
        )
        logger = logging.getLogger(__name__)

        # Build the network and load weights
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dummy_dataset)
        self.model.load_params_from_file(filename=CKPT_FILE, logger=logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

    
    def decodeImage(self, channel, proto_msg):
        print("image: ", proto_msg.width, "x", proto_msg.height, "type: ", proto_msg.type) 
        # JPEG
        if(proto_msg.type == 10):
            # Convert the bytes into a NumPy uint8 array
            nparr = np.frombuffer(proto_msg.data, np.uint8)

            # Decode the array into an image (OpenCV format)
            self.img = None
            if proto_msg.channels == 3 or proto_msg.channels == 4:
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif proto_msg.channels == 1:
                self.img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:
                raise TypeError("Unsupported channels: ", proto_msg.channels)

            # debug viz
            # cv2.imshow(channel.topic, self.img)
            # cv2.waitKey(1)   
        elif(proto_msg.type == 0):
            # Convert the bytes into a NumPy uint8 array
            nparr = np.frombuffer(proto_msg.data, np.uint8)
            # cast array to mat
            self.img = nparr.reshape((proto_msg.height, proto_msg.width, proto_msg.channels))

            # debug viz
            cv2.imshow(channel.topic, self.img)
            cv2.waitKey(1)
        else:
            raise TypeError("Unsupported image type: ", proto_msg.type)
        
    def decodeOdometry(self, proto_msg):
        #self.shift_x = proto_msg.twist.linear.x
        #self.shift_y = proto_msg.twist.linear.y

        self.shift_x = -proto_msg.twist.linear.y
        self.shift_y = proto_msg.twist.linear.x

        self.shift_yaw = proto_msg.twist.angular.z

    def decodeCloud(self, data):
        self.points = protoCloudToNumpy(data)
        print("Pointcloud: ", np.shape(self.points))

        # debug viz
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # o3d.visualization.draw_geometries([pcd])

    def processCloud(self):

        points_with_time = self.processPoints()
        
        pred_dicts, recall_dicts = self.runInference(points_with_time)
        

        predictions = pred_dicts[0]

        # Move results back to CPU and convert to NumPy
        pred_boxes = predictions['pred_boxes'].cpu().numpy()
        pred_scores = predictions['pred_scores'].cpu().numpy()
        pred_labels = predictions['pred_labels'].cpu().numpy()

        # ==========================================
        # 5. VIEW RESULTS
        # ==========================================
        # OpenPCDet labels are usually 1-indexed (1: Car, 2: Pedestrian, 3: Cyclist)
        class_names = cfg.CLASS_NAMES

        print(f"Found {len(pred_boxes)} total objects.")
        for i in range(len(pred_boxes)):
            if pred_scores[i] > 0.3:  # Only print confident predictions
                class_name = class_names[pred_labels[i] - 1]
                print(f"[{class_name}] Confidence: {pred_scores[i]:.2f} | Box (x,y,z,l,w,h,theta): {pred_boxes[i]}")

        # print("points shape: ", np.shape(self.points))
        # print("predictions: ", predictions)
        # print("pred_boxes shape: ", np.shape(pred_boxes))
        # print("pred_scores shape: ", np.shape(pred_scores))
        # print("pred_labels shape: ", np.shape(pred_labels))
        # print("img shape: ", np.shape(self.img))
        
        # ------------ BEV Visualization -------------
        
        color_dict = {}
        for i, v in enumerate(cfg.CLASS_NAMES):
            # Applied the .get() fallback to prevent KeyErrors for missing classes
            color_dict[v] = label_color_palette_2d.get(v, (128, 128, 128))
        saveODImgs(anno=predictions, pts=self.points, img_path=IMG_PATH_BEV, color_dict=color_dict, title='pred', fid=self.counter)
        self.counter += 1
        # -------------------------------------------

        self.visualize(predictions)

    def calculate_compensated_velocity(self, points):
        """
        Calculates the absolute compensated radial velocity for radar points.
        
        Args:
            points: (N, 7) numpy array where columns are [x, y, z, RCS, max_v_r, v_r, v_r_comp, time]
        
        Returns:
            v_comp: (N,) numpy array of compensated velocities
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        v_meas = points[:, 6] # The raw Doppler velocity from the radar
        

        v_x = np.sqrt(self.shift_x**2 + self.shift_y**2)  
        v_y, omega_z = 0, self.shift_yaw
        # v_x, v_y, omega_z = self.shift_x, self.shift_y, self.shift_yaw
        t_x, t_y, yaw = RADAR_OFFSET_TX, RADAR_OFFSET_TY, RADAR_OFFSET_YAW  
        
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

        print("EGO VELOCITY (X, Y, YAW): ", v_x, v_y, omega_z)
        print("MEAN RADIAL VELOCITY BEFORE COMPENSATION: ", np.mean(v_meas))
        print("MEAN EGO LOS VELOCITY: ", np.mean(v_ego_los))
        print("MEAN RADIAL VELOCITY AFTER COMPENSATION: ", np.mean(v_comp))
        
        return v_comp
    
    def processPoints(self):
        v_comp = self.calculate_compensated_velocity(self.points)

        radial_ambiguous_velocity = np.expand_dims(self.points[:,6], axis=1)
        #print("Speed: ", np.shape(radial_ambiguous_velocity))
        v_comp=np.expand_dims(v_comp, axis=1)

        snr = np.expand_dims(self.points[:,4], axis=1)

        time_vector = np.zeros((self.points.shape[0], 1), dtype=self.points.dtype)
        self.points = np.hstack([self.points[:, 0:3], snr, radial_ambiguous_velocity, v_comp, time_vector])
        
        print("Points with batch: ", np.shape(self.points))

        return self.points
    
    def runInference(self, points_with_time):
        input_dict = {
            'points': points_with_time,
            'frame_id': 1,
        }

        # Let the dummy dataset automatically voxelize the points based on your YAML!
        data_dict = self.dummy_dataset.prepare_data(data_dict=input_dict)

        batch_dict = DatasetTemplate.collate_batch([data_dict])

        # Move everything to the GPU
        for key, val in batch_dict.items():
            if isinstance(val, torch.Tensor):
                batch_dict[key] = val.cuda()
            elif isinstance(val, np.ndarray):
                batch_dict[key] = torch.from_numpy(val).cuda()


        with torch.no_grad():
            pred_dicts, recall_dicts = self.model(batch_dict=batch_dict)
        
        return pred_dicts, recall_dicts

    def visualize(self, predictions):
        
        # 4. Generate the image!
        self.vis.draw_plot(
            img=self.img,
            points=self.points,
            predictions=predictions,
            save_figure=True, 
            show_pred=True, 
            show_radar=True, 
            score_threshold=0.3
        )



if __name__ == "__main__":
    processor = DataProcessor()
    with open(sys.argv[1], "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
            if(schema.name == "proto.tk.msg.Image"):
                #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                processor.decodeImage(channel, proto_msg)
            elif(schema.name == "proto.tk.msg.Cloud"):
                if processor.img is not None and channel.topic == "/radar/cloud/radar_fc":
                    print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                    processor.decodeCloud(proto_msg)
                    processor.processCloud()
                
                #print(f"msg {proto_msg.type}]")


                # Marzaglia (1 radar):
                # if processor.img is not None:
                #     processor.decodeCloud(proto_msg)
                #     processor.processCloud()
            elif(schema.name == "proto.tk.msg.Odometry"):
                processor.decodeOdometry(proto_msg)