# Format: 
# [ f_x,   0.0,  c_x ]
# [ 0.0,   f_y,  c_y ]
# [ 0.0,   0.0,  1.0 ]

NET = "PP"  
#NET = "MM-FA"

#LOG = "4PORTE"
#LOG = "MARZAGLIA"
LOG = "NEW_RADAR_4PORTE"


#DATASET = "NUSCENES"  
#DATASET = "VOD"  
DATASET = "TRUCKSCENES"

DIVIDED_CARS = True  # Whether to divide cars into moving and stopped classes

N_FRAMES = 10

                            



# MODEL SETTINGS
if NET == "PP":
    BASE_DATA_PATH = "/media/franco/hdd/dataset/radar_data/"
    OUTPUT_DIR_IMGS_2D = "./results/2D/images"
    IMG_PATH_BEV = "/media/franco/hdd/matteogombia/OpenPCDet/tools/results/BEV/images"

    # YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte.yaml"

    if DATASET == "VOD":
        if N_FRAMES == 5:
            #CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar.yaml"
            #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar/default/ckpt/checkpoint_epoch_125.pth" 
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar_noz.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar_noz/default/ckpt/checkpoint_epoch_80.pth"

        else:
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/kitti_models/PP_radar_1frame.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/kitti_models/PP_radar_1frame/default/ckpt/checkpoint_epoch_80.pth"

    # NuScenes
    if DATASET == "NUSCENES":
        if DIVIDED_CARS == False:
            #CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/PP_nuscenes_radar.yaml"
            #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_40.pth" 
            #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar_old/default/ckpt/checkpoint_epoch_40.pth"
            #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_100.pth"
            #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_100.pth"
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/cbgs_pp_multihead_radar_nostride/default/ckpt/checkpoint_epoch_100.pth"
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead_radar_nostride.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/cbgs_pp_multihead_radar_nostride_30ep/default/ckpt/checkpoint_epoch_30.pth"
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead_radar_nostride_30ep.yaml"
        else:
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_cars_models/cbgs_pp_multihead_radar_nostride_30ep/default/ckpt/checkpoint_epoch_30.pth"
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_cars_models/cbgs_pp_multihead_radar_nostride_30ep.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_cars_models/cbgs_pp_multihead_radar_nostride_30ep_norm/default/ckpt/checkpoint_epoch_26.pth"
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_cars_models/cbgs_pp_multihead_radar_nostride_30ep_norm.yaml"
            

    if DATASET == "TRUCKSCENES":
        if DIVIDED_CARS == False:
            #CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_models/cbgs_pp_multihead_radar_vcomp/default/cbgs_pp_multihead_radar_vcomp.yaml"
            #CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_models/cbgs_pp_multihead_radar_vcomp/default/ckpt/checkpoint_epoch_30.pth"
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_models/cbgs_pp_multihead_radar_vcomp_neg04/default/cbgs_pp_multihead_radar_vcomp_neg04.yaml"
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_models/cbgs_pp_multihead_radar_vcomp_neg04/default/ckpt/checkpoint_epoch_12.pth"
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_models/cbgs_pp_multihead_radar_vcomp_nostride/default/cbgs_pp_multihead_radar_vcomp_nostride.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_models/cbgs_pp_multihead_radar_vcomp_nostride/default/ckpt/checkpoint_epoch_30.pth"
        else:
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pp_multihead_radar_vcomp_nostride/default/cbgs_pp_multihead_radar_vcomp_nostride.yaml"
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pp_multihead_radar_vcomp_nostride/default/ckpt/checkpoint_epoch_23.pth"
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pp_multihead_radar_vcomp_nostride_attr/default/cbgs_pp_multihead_radar_vcomp_nostride_attr.yaml"
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pp_multihead_radar_vcomp_nostride_attr/default/ckpt/checkpoint_epoch_23.pth"
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pillar0075_res2d_centerpoint/default/cbgs_pillar0075_res2d_centerpoint.yaml"
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pillar0075_res2d_centerpoint/default/ckpt/checkpoint_epoch_30.pth"
            CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pp_multihead_radar_vcomp_nostride_attr_16bs/default/cbgs_pp_multihead_radar_vcomp_nostride_attr_16bs_range300.yaml"
            CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pp_multihead_radar_vcomp_nostride_attr_16bs/default/ckpt/checkpoint_epoch_21.pth"
            # CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pillar0075_res2d_centerpoint_8bs/default/cbgs_pillar0075_res2d_centerpoint_8bs_range300.yaml"
            # CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/truckscenes_multiradar_cars_models/cbgs_pillar0075_res2d_centerpoint_8bs/default/ckpt/checkpoint_epoch_30.pth"
elif NET == "MM-FA":
    BASE_DATA_PATH = "/root/data/"
    OUTPUT_DIR_IMGS_2D = "/seeing_beyond/tools/results/2D/images"
    IMG_PATH_BEV = "/seeing_beyond/tools/results/BEV/images"

    # YAML_PATH="/media/franco/hdd/dataset/radar_data/calib_4porte.yaml"

    if DATASET == "VOD":
        
        if N_FRAMES == 1:
            CKPT_FILE = "/seeing_beyond/ckpts/cfar/cfar-radar.pth" 
            CFG_FILE = "/seeing_beyond/tools/cfgs/kitti_models/cfar-radar.yaml"
        else:
            CKPT_FILE = "/seeing_beyond/output/cfar-radar-5frames/debug_new/ckpt/checkpoint_epoch_40.pth"
            CFG_FILE = "/seeing_beyond/tools/cfgs/kitti_models/cfar-radar-5frames.yaml"

    # NuScenes
    if DATASET == "NUSCENES":
        CFG_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/tools/cfgs/nuscenes_models/PP_nuscenes_radar.yaml"
        CKPT_FILE = "/media/franco/hdd/matteogombia/OpenPCDet/output/nuscenes_models/PP_nuscenes_radar/default/ckpt/checkpoint_epoch_40.pth" 



# ------------ LOG SETTINGS ----------------
# # Marzaglia
if LOG == "MARZAGLIA":
    
    radar_frame_id = "radar_fc"
    camera_frame_id = "camera_fcl"
    YAML_PATH=BASE_DATA_PATH + "calib_4porte_marzaglia.yaml"
    # For Marzaglia, the camera intrinsics are actually not precise
    my_camera_intrinsics = [
        [1200.0,    0.0,  960.0],
        [   0.0, 1200.0,  540.0],
        [   0.0,    0.0,    1.0]
    ]
    MCAP_PATH = BASE_DATA_PATH + "marzaglia_with_odom.mcap"
    RADAR_OFFSET_TX = 4.0  # meters forward from the vehicle's center
    RADAR_OFFSET_TY = 0.0  # meters to the left of the vehicle
    RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

    RADAR_RX_OFFSET_X = 0.0
    RADAR_RX_OFFSET_Y = -0.5
    RADAR_RX_OFFSET_ANGLE_DEG = -32.0

    RADAR_LX_OFFSET_X = 0.0
    RADAR_LX_OFFSET_Y = 0.5
    RADAR_LX_OFFSET_ANGLE_DEG = 32.5

    IS_RCS_NORMALIZED = True  # Whether the RCS values are normalized (0-1) or in dBsm

# # 4porte
if LOG == "4PORTE":
    radar_frame_id = "radar_fc"
    camera_frame_id = "cam_f"
    YAML_PATH=BASE_DATA_PATH + "calib_4porte.yaml"
    # For 4porte, the camera intrinsics are actually not precise
    my_camera_intrinsics = [
        [800.0,   0.0, 400.0],
        [0.0, 800.0, 300.0],
        [0.0,   0.0,   1.0]
    ]
    MCAP_PATH = BASE_DATA_PATH + "quattroporte_hipert_with_odom.mcap"
    RADAR_OFFSET_TX = 3.5  # meters forward from the vehicle's center
    RADAR_OFFSET_TY = -0.5  # meters to the left of the vehicle
    RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

    RADAR_RX_OFFSET_X = 0.0
    RADAR_RX_OFFSET_Y = -0.5
    RADAR_RX_OFFSET_ANGLE_DEG = -32.0

    RADAR_LX_OFFSET_X = 0.0
    RADAR_LX_OFFSET_Y = 0.5
    RADAR_LX_OFFSET_ANGLE_DEG = 32.5

    IS_RCS_NORMALIZED = True  # Whether the RCS values are normalized (0-1) or in dBsm

if LOG == "NEW_RADAR_4PORTE":
    radar_frame_id = "radar_fc"
    camera_frame_id = "cam_f"
    YAML_PATH= BASE_DATA_PATH + "newradar_calib_4porte.yaml"
    # For 4porte, the camera intrinsics are actually not precise
    my_camera_intrinsics = [
        [1171.6981821866186,   0.0, 943.35662623835799],
        [0.0, 1167.6873973411323, 517.18535312115455],
        [0.0,   0.0,   1.0]
    ]
    MCAP_PATH = BASE_DATA_PATH + "newradar_4porte_modena_with_cam.mcap"
    RADAR_OFFSET_TX = 4.3  # meters forward from the vehicle's center
    RADAR_OFFSET_TY = 0.0  # meters to the left of the vehicle
    RADAR_OFFSET_YAW = 0.0  # radians, assuming radar faces forward with no rotation

    RADAR_RX_OFFSET_X = 0.0
    RADAR_RX_OFFSET_Y = -0.7
    RADAR_RX_OFFSET_ANGLE_DEG = -17.5

    RADAR_LX_OFFSET_X = 0.0
    RADAR_LX_OFFSET_Y = 0.7
    RADAR_LX_OFFSET_ANGLE_DEG = 19.0

    IS_RCS_NORMALIZED = False  # Whether the RCS values are normalized (0-1) or in dBsm