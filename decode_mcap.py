import sys
import numpy as np
from point_processor_nuscenes import PointProcessorNuscenes
from point_processor import PointProcessor
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

LOG = "4PORTE"
# LOG = "MARZAGLIA"

#DATASET = "NUSCENES"  
DATASET = "VOD"  




# # Marzaglia
if LOG == "MARZAGLIA":
    radar_frame_id = "radar_fc"
    camera_frame_id = "camera_fcl"
    YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte_marzaglia.yaml"
    # For Marzaglia, the camera intrinsics are actually not precise
    my_camera_intrinsics = [
        [1200.0,    0.0,  960.0],
        [   0.0, 1200.0,  540.0],
        [   0.0,    0.0,    1.0]
    ]
    MCAP_PATH = "/media/franco/hdd/dataset/radar_data/marzaglia_with_odom.mcap"

# # 4porte
if LOG == "4PORTE":
    radar_frame_id = "radar_fc"
    camera_frame_id = "cam_f"
    YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte.yaml"
    # For 4porte, the camera intrinsics are actually not precise
    my_camera_intrinsics = [
        [800.0,   0.0, 400.0],
        [0.0, 800.0, 300.0],
        [0.0,   0.0,   1.0]
    ]
    MCAP_PATH = "/media/franco/hdd/dataset/radar_data/quattroporte_hipert_with_odom.mcap"

RADAR_OFFSET_TX = 3.5  # meters forward from the vehicle's center
RADAR_OFFSET_TY = -0.5  # meters to the left of the vehicle
RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

OUTPUT_DIR_IMGS_2D = "./results/2D/images"
IMG_PATH_BEV = "/media/franco/hdd/matteogombia/OpenPCDet/tools/results/BEV/images"

# YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte.yaml"

if DATASET == "VOD":
    CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar.yaml"
    CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar/default/ckpt/checkpoint_epoch_125.pth" 

# NuScenes
if DATASET == "NUSCENES":
    CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/PP_nuscenes_radar.yaml"
    CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_28.pth" 


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
        self.cloud = None
        self.counter = 0

        if DATASET == "VOD":
            self.points_processor = PointProcessor(
                radar_offset_tx=RADAR_OFFSET_TX,
                radar_offset_ty=RADAR_OFFSET_TY,
                radar_offset_yaw=RADAR_OFFSET_YAW,
                n_frames=5
            )
        if DATASET == "NUSCENES":
            self.points_processor = PointProcessorNuscenes(
                radar_offset_tx=RADAR_OFFSET_TX,
                radar_offset_ty=RADAR_OFFSET_TY,
                radar_offset_yaw=RADAR_OFFSET_YAW,
                n_frames=6
            )

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
            self.points_processor.img = None
            if proto_msg.channels == 3 or proto_msg.channels == 4:
                self.points_processor.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif proto_msg.channels == 1:
                self.points_processor.img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:
                raise TypeError("Unsupported channels: ", proto_msg.channels)

            # debug viz
            # cv2.imshow(channel.topic, self.points_processor.img)
            # cv2.waitKey(1)   
        elif(proto_msg.type == 0):
            # Convert the bytes into a NumPy uint8 array
            nparr = np.frombuffer(proto_msg.data, np.uint8)
            # cast array to mat
            self.points_processor.img = nparr.reshape((proto_msg.height, proto_msg.width, proto_msg.channels))

            # debug viz
            cv2.imshow(channel.topic, self.points_processor.img)
            cv2.waitKey(1)
        else:
            raise TypeError("Unsupported image type: ", proto_msg.type)
        
    def decodeOdometry(self, proto_msg):
        self.points_processor.shift_x = proto_msg.twist.linear.x
        self.points_processor.shift_y = proto_msg.twist.linear.y
         
        self.points_processor.shift_yaw = proto_msg.twist.angular.z

    def decodeCloud(self, data):
        self.points = protoCloudToNumpy(data)
        print("Pointcloud: ", np.shape(self.points))

        # debug viz
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # o3d.visualization.draw_geometries([pcd])

    def processCloud(self):
        points_multiframe = self.points_processor.processPoints(self.points)
        
        pred_dicts, recall_dicts = self.runInference(points_multiframe)
        

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
        # print("img shape: ", np.shape(self.points_processor.img))
        
        # ------------ BEV Visualization -------------
        
        color_dict = {}
        for i, v in enumerate(cfg.CLASS_NAMES):
            if v == 'bicycle':
                v = 'Cyclist' 
            if v == 'pedestrian':
                v = 'Pedestrian'
            if v == 'car':
                v = 'Car'
            # Applied the .get() fallback to prevent KeyErrors for missing classes
            color_dict[v] = label_color_palette_2d.get(v, (128, 128, 128))

        print(color_dict)
        saveODImgs(anno=predictions, pts=points_multiframe, img_path=IMG_PATH_BEV, color_dict=color_dict, title='pred', fid=self.counter)
        self.counter += 1
        # -------------------------------------------

        self.visualize(points_multiframe, predictions)

    
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

    def visualize(self, points, predictions):
        
        # 4. Generate the image!
        self.vis.draw_plot(
            img=self.points_processor.img,
            points=points,
            predictions=predictions,
            save_figure=True, 
            show_pred=True, 
            show_radar=True, 
            score_threshold=0.4
        )



if __name__ == "__main__":
    processor = DataProcessor()
    i = 0
    counter_odom = 0
    counter_cloud = 0
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
            if(schema.name == "proto.tk.msg.Image"):
                #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                processor.decodeImage(channel, proto_msg)

            elif(schema.name == "proto.tk.msg.Cloud"):
                if processor.points_processor.img is not None and channel.topic == "/radar/cloud/radar_fc" and i > 400:
                    #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                    # processor.decodeCloud(proto_msg)
                    # processor.processCloud()
                    counter_cloud += 1
                    if counter_cloud % 50 == 0:
                        print(f"Processed {counter_cloud} radar frames, total odometry messages: {counter_odom}")
                else: 
                    i += 1
                    counter_cloud += 1

                
                #print(f"msg {proto_msg.type}]")


                # Marzaglia (1 radar):
                # if processor.img is not None:
                #     processor.decodeCloud(proto_msg)
                #     processor.processCloud()
            # elif(schema.name == "proto.tk.msg.Odometry"):
            #     processor.decodeOdometry(proto_msg)
            elif (channel.topic == "/odom/debug"):
                processor.decodeOdometry(proto_msg)
                print(f"msg {proto_msg}]")
                counter_odom += 1