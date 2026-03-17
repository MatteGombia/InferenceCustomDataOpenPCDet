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
from config import *

import cv2


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
        self.left_points = None
        self.right_points = None

        if DATASET == "VOD":
            self.points_processor = PointProcessor(
                radar_offset_tx=RADAR_OFFSET_TX,
                radar_offset_ty=RADAR_OFFSET_TY,
                radar_offset_yaw=RADAR_OFFSET_YAW,
                n_frames=N_FRAMES
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
        #print("image: ", proto_msg.width, "x", proto_msg.height, "type: ", proto_msg.type) 
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
        self.points_processor.add_odometry(proto_msg.twist.linear.x, proto_msg.twist.linear.y, proto_msg.twist.angular.z, proto_msg.head.stamp)

    def decodeCloud(self, data):
        if data.head.frameId == radar_frame_id:
            self.points = protoCloudToNumpy(data)
            print("Pointcloud: ", np.shape(self.points))
            self.points_processor.new_pc_arrived = True
       
            self.points_processor.add_timestamp(data.head.stamp)

        elif data.head.frameId == "radar_fl":
            self.left_points = protoCloudToNumpy(data)
            self.points_processor.timestamp_last_frame_left = data.head.stamp

        elif data.head.frameId == "radar_fr":
            self.right_points = protoCloudToNumpy(data)
            self.points_processor.timestamp_last_frame_right = data.head.stamp


    def processCloud(self):
        points_multiframe = self.points_processor.processPoints(self.points)
        
        if self.left_points is not None:
            print("Adding left radar points to multiframe processor, timestamp diff: ", (self.points_processor.timestamp_last_frame_left-self.points_processor.timestamp_last_frame)*1e-9)
            self.points_processor.add_auxiliar_cloud(self.left_points, self.points_processor.timestamp_last_frame_left, 0.0, 0.5, 32.5)
        if self.right_points is not None:
            print("Adding right radar points to multiframe processor, timestamp diff: ", (self.points_processor.timestamp_last_frame_right-self.points_processor.timestamp_last_frame)*1e-9)
            self.points_processor.add_auxiliar_cloud(self.right_points, self.points_processor.timestamp_last_frame_right, 0.0, -0.5, -32.0)

        points_multiframe = self.points_processor.multiframe_points
        
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

        # print(color_dict)
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
        
        # Generate the image
        self.vis.draw_plot(
            img=self.points_processor.img,
            points=points,
            predictions=predictions,
            save_figure=True, 
            show_pred=True, 
            show_radar=True, 
            score_threshold=0.5
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
                #print(f"msg {proto_msg}]")
                if processor.points_processor.img is not None:
                    if channel.topic == "/radar/cloud/radar_fc" and i > 650:
                        #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                        #print(f"msg {proto_msg}]")
                        processor.decodeCloud(proto_msg)
                        counter_cloud += 1
                        # if counter_cloud % 2000 == 0:
                        #     processor.points_processor.print_bins()
                        #     sys.exit(0)
                        if counter_cloud % 50 == 0:
                            print(f"Processed {counter_cloud} radar frames, total odometry messages: {counter_odom}")
                    elif channel.topic == "/radar/cloud/radar_fr" and i > 650:
                        #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                        processor.decodeCloud(proto_msg)
                    elif channel.topic == "/radar/cloud/radar_fl" and i > 650:
                        #print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
                        processor.decodeCloud(proto_msg)
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
                #print(f"msg {proto_msg}]")
                processor.decodeOdometry(proto_msg)
                counter_odom += 1

                if processor.points_processor.new_pc_arrived:
                    processor.processCloud()