# Format: 
# [ f_x,   0.0,  c_x ]
# [ 0.0,   f_y,  c_y ]
# [ 0.0,   0.0,  1.0 ]

NET = "PP"  
#NET = "MM-FA"

LOG = "4PORTE"
#LOG = "MARZAGLIA"

#DATASET = "NUSCENES"  
DATASET = "VOD"  

N_FRAMES = 5

if NET == "PP":
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
        RADAR_OFFSET_TX = 4.0  # meters forward from the vehicle's center
        RADAR_OFFSET_TY = 0.0  # meters to the left of the vehicle
        RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

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
        if N_FRAMES == 5:
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar/default/ckpt/checkpoint_epoch_125.pth" 
        else:
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar_1frame.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar_1frame/default/ckpt/checkpoint_epoch_80.pth"

    # NuScenes
    if DATASET == "NUSCENES":
        CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/PP_nuscenes_radar.yaml"
        #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_40.pth" 
        #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar_old/default/ckpt/checkpoint_epoch_40.pth"
        CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_100.pth"

elif NET == "MM-FA":
    if LOG == "MARZAGLIA":
        radar_frame_id = "radar_fc"
        camera_frame_id = "camera_fcl"
        YAML_PATH="/root/data/calib_4porte_marzaglia.yaml"
        # For Marzaglia, the camera intrinsics are actually not precise
        my_camera_intrinsics = [
            [1200.0,    0.0,  960.0],
            [   0.0, 1200.0,  540.0],
            [   0.0,    0.0,    1.0]
        ]
        MCAP_PATH = "/root/data/marzaglia_with_odom.mcap"
        RADAR_OFFSET_TX = 4.0  # meters forward from the vehicle's center
        RADAR_OFFSET_TY = 0.0  # meters to the left of the vehicle
        RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

    # # 4porte
    if LOG == "4PORTE":
        radar_frame_id = "radar_fc"
        camera_frame_id = "cam_f"
        YAML_PATH="/root/data/calib_4porte.yaml"
        # For 4porte, the camera intrinsics are actually not precise
        my_camera_intrinsics = [
            [800.0,   0.0, 400.0],
            [0.0, 800.0, 300.0],
            [0.0,   0.0,   1.0]
        ]
        MCAP_PATH = "/root/data/quattroporte_hipert_with_odom.mcap"
        RADAR_OFFSET_TX = 3.5  # meters forward from the vehicle's center
        RADAR_OFFSET_TY = -0.5  # meters to the left of the vehicle
        RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation



    OUTPUT_DIR_IMGS_2D = "/seeing_beyond/tools/results/2D/images"
    IMG_PATH_BEV = "/seeing_beyond/tools/results/BEV/images"

    # YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte.yaml"

    if DATASET == "VOD":
        CFG_FILE = "/seeing_beyond/tools/cfgs/kitti_models/cfar-radar.yaml"
        if N_FRAMES == 1:
            CKPT_FILE = "/seeing_beyond/ckpts/cfar/cfar-radar.pth" 
        else:
            CKPT_FILE = "/seeing_beyond/output/cfar-radar-5frames/debug_new/ckpt/checkpoint_epoch_36.pth"

    # NuScenes
    if DATASET == "NUSCENES":
        CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/PP_nuscenes_radar.yaml"
        CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_40.pth" 

