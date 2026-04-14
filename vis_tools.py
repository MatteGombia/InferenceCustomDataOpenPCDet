import numpy as np
from pathlib import Path as P
import numpy as np
from pcdet.utils import calibration_kitti, object3d_kitti
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from skimage import io
from matplotlib.transforms import Affine2D
    
MIN_SCORE_THRESHOLD = 0.3

def drawBEV(ax, pts, annos, color_dict, ax_title, cls_names, ext_legends=[]):
    
    pred_boxes = annos['pred_boxes'].cpu().numpy()
    pred_scores = annos['pred_scores'].cpu().numpy()
    pred_labels = annos['pred_labels'].cpu().numpy()

    # 1. draw original points if exist
    if pts is not None:
        #speed = np.abs(pts[:, -2])
        #pts = pts[np.sqrt(pts[:, -3]**2 + pts[:, -2]**2) > 1]
        speed = np.sqrt(pts[:, -3]**2 + pts[:, -2]**2)  
        
        x = pts[:, 0]
        y = pts[:, 1]
        print("Max and min speed in points: ", np.max(speed), np.min(speed))
        #ax.scatter(x, y, c='black', s=0.1)
        #ax.scatter(x, y, c=pts[:, -1], s=0.1)
        ax.scatter(x, y, c=speed, s=0.1)
        #lat_speed = np.fmin(np.abs(pts[:, -3]), 2)
        #ax.scatter(x, y, c=lat_speed, s=0.1)


        # Check for speed direction
        # pts = pts[pts[:, -1]==0]
        # for pt in pts:
        
        #     #ax.annotate("", xytext=(pt[0],pt[1]), xy=(pt[-3], pt[-2]), arrowprops=dict(arrowstyle="->"))
        #     dx = pt[-3]
        #     dy = pt[-2]
        #     ax.arrow(pt[0],pt[1],dx/2, dy/2, head_width=1.5, width=0.0,  alpha=0.5, edgecolor="#000000", fc="#000000")



    # --- Confidence Scores ---
    # Handle the fact that annos might be a dictionary or a list containing a dictionary
    
    for i in range(len(pred_scores)):
        
        score = pred_scores[i]
        rec = pred_boxes[i] if i < len(pred_boxes) else None
        # Only draw scores for predictions (Ground Truths usually lack scores or have dummy 0/-1 values)
        if score > MIN_SCORE_THRESHOLD and rec is not None:
            
            x_center = rec[0]
            y_center = rec[1]
            l = rec[3]
            w = rec[4]
            ang = rec[6] # In radians from the network

            # Calculate anchor point (bottom-left corner) for the Rectangle patch
            x_bottom_left = x_center - (l / 2) * np.cos(ang) + (w / 2) * np.sin(ang)
            y_bottom_left = y_center - (l / 2) * np.sin(ang) - (w / 2) * np.cos(ang)

            ang_deg = ang * 180 / np.pi  

            label = pred_labels[i]
            color = color_dict[cls_names[label-1]] if label-1 < len(cls_names) else (128, 128, 128)

            rec_patch = Rec(
                (x_bottom_left, y_bottom_left), 
                l, 
                w, 
                angle=ang_deg, 
                fill=False, 
                color=color,
                lw=2
            )
            ax.add_patch(rec_patch)

            # Draw a line from the center to the front of the box to indicate orientation
            corners = rec_patch.get_corners()
            #print(f"Box corners: {corners}")


            center_bottom_forward = np.mean(corners[1:3], axis=0)
            ax.plot([x_center, center_bottom_forward[0]],
                        [y_center, center_bottom_forward[1]],
                        color=color, linewidth=2)

            # ------------------------------------------
            


            # Draw the text with a small black background so it's easy to read over point clouds
            ax.text(float(max([corner[0] for corner in corners])), float(max([corner[1] for corner in corners])), f"{score:.2f}", color='b', fontsize=7, 
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=0.5))
    # -----------------------------------

    legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]

    legend_elements += ext_legends
    
    #ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)

def saveODImgs(anno, pts, img_path, color_dict, cls_names, title='pred', fid = 0):
    #print('=================== drawing images ===================')
    plt.rcParams['figure.dpi'] = 150
   
    ax = plt.gca()
    drawBEV(ax, pts, anno, color_dict, title, cls_names)
    
    img_fname = P(f"{img_path}/{fid}.png")
    plt.xlim(0, 150)
    #plt.xlim(0, 300)
    plt.ylim(-50, 50)
    plt.savefig(str(img_fname))
    plt.cla()
