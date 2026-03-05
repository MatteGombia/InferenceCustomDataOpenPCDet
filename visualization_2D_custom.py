import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pcdet.utils import box_utils

class Visualization2D:
    """
    A standalone class to visualize an image overlaid with 3D bounding boxes 
    and radar/LiDAR point clouds using OpenPCDet calibration.
    """
    def __init__(self, 
                 calib,  
                 output_dir: str = './output/', 
                 classes_visualized: list = [1, 2, 3]): # 1: Car, 2: Pedestrian, 3: Cyclist
        
        self.calib = calib
        self.output_dir = output_dir
        self.classes_visualized = classes_visualized
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.counter = 0

        # Map OpenPCDet integer labels to colors (RGB 0-1)
        self.color_map = {
            1: (0.2, 0.6, 1.0),       # 1 = Car (Blue)
            2: (0.2, 1.0, 0.2),       # 2 = Pedestrian (Green)
            3: (1.0, 0.2, 0.2)      # 3 = Cyclist (Red)
        }

    def plot_predictions(self, ax, score_threshold: float):
        """
        Filters the predictions by score and class, projects the 3D corners 
        onto the 2D image plane, and draws the translucent box faces.
        """
        # FIX 1: Check for 'pred_boxes' instead of 'name'
        if self.predictions is None or 'pred_boxes' not in self.predictions:
            return

        scores = self.predictions['pred_scores'].cpu().numpy()
        labels = self.predictions['pred_labels'].cpu().numpy()
        boxes = self.predictions['pred_boxes'].cpu().numpy() 

        # FIX 2: Filter using the integer labels
        mask = (scores > score_threshold) & np.isin(labels, self.classes_visualized)
        
        valid_boxes = boxes[mask]
        valid_labels = labels[mask]

        if len(valid_boxes) == 0:
            return

        # Convert to 8 3D corners: (N, 8, 3)
        boxes_tensor = torch.tensor(valid_boxes)
        corners_3d = box_utils.boxes_to_corners_3d(boxes_tensor).numpy()

        # The 5 faces we want to draw (Front, Right, Back, Left, Top)
        faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]

        for i, corners in enumerate(corners_3d):
            # Now it correctly looks up the integer (e.g., 1) to get Red
            color = self.color_map.get(valid_labels[i], (1, 1, 1))

            # Project 3D corners to 2D pixels
            corners_2d, depths = self.calib.lidar_to_img(corners)

            # Only draw if at least one corner is in front of the camera (depth > 0)
            if np.any(depths > 0):
                for face in faces:
                    poly = patches.Polygon(corners_2d[face], closed=True, 
                                           facecolor=color, alpha=0.3, 
                                           edgecolor=color, linewidth=1.5)
                    ax.add_patch(poly)

    def plot_radar_pcl(self, ax, min_distance_threshold: float, max_distance_threshold: float):
        if self.points is None or len(self.points) == 0:
            return

        H, W = self.img.shape[:2]
        pts = self.points.cpu().numpy() if torch.is_tensor(self.points) else self.points

        # Because your data is [x, y, z, rcs, velocity, v_comp], we just grab columns 0,1,2
        pts_img, pts_depth = self.calib.lidar_to_img(pts[:, :3])

        print(f"--- PROJECTION DEBUG ---")
        print(f"Image Resolution: {W}x{H}")
        print(f"Min/Max Depth: {np.min(pts_depth):.2f}m to {np.max(pts_depth):.2f}m")
        print(f"U (X-Pixels): {np.min(pts_img[:, 0]):.1f} to {np.max(pts_img[:, 0]):.1f}")
        print(f"V (Y-Pixels): {np.min(pts_img[:, 1]):.1f} to {np.max(pts_img[:, 1]):.1f}")
        print(f"------------------------")

        mask = (pts_depth > min_distance_threshold) & (pts_depth < max_distance_threshold) & \
               (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) & \
               (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H)

        valid_pts_img = pts_img[mask]
        valid_pts_depth = pts_depth[mask]
        print("Valid radar points after filtering: ", valid_pts_img.shape[0])
        ax.scatter(valid_pts_img[:, 0], valid_pts_img[:, 1], 
                   c=valid_pts_depth, cmap='jet', s=30, alpha=0.9, zorder=10)

    def draw_plot(self, img: np.ndarray, 
                  points: np.ndarray, 
                  predictions: dict, 
                  save_figure: bool = True, filename: str = None,
                  show_pred: bool = True, show_radar: bool = True,
                  max_distance_threshold: float = 120.0, min_distance_threshold: float = 0.0,
                  score_threshold: float = 0.3):
        """
        Renders the final image with all requested overlays.
        """
        self.img = img 
        self.points = points
        self.predictions = predictions
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

        H, W = self.img.shape[:2]

        ax.imshow(self.img)

        # Lock the axes to the image size
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis('off')

        if show_radar:
            self.plot_radar_pcl(ax, min_distance_threshold, max_distance_threshold)

        if show_pred:
            self.plot_predictions(ax, score_threshold)

        if save_figure:
            out_name = filename if filename else f"frame_{self.counter}.png"
            out_path = os.path.join(self.output_dir, out_name)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            self.counter += 1
            
        plt.close(fig)