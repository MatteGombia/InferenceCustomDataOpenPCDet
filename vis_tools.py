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
    

def drawBEV(ax, pts, annos, color_dict, ax_title, ext_legends=[]):
    
    pred_boxes = annos['pred_boxes'].cpu().numpy()
    pred_scores = annos['pred_scores'].cpu().numpy()
    pred_labels = annos['pred_labels'].cpu().numpy()

    # 1. draw original points if exist
    if pts is not None:
        x = pts[:, 0]
        y = pts[:, 1]
        #ax.scatter(x, y, c='black', s=0.1)
        ax.scatter(x, y, c=pts[:, -1], s=0.1)


    # --- Confidence Scores ---
    # Handle the fact that annos might be a dictionary or a list containing a dictionary
    
    for i in range(len(pred_scores)):
        
        score = pred_scores[i]
        rec = pred_boxes[i] if i < len(pred_boxes) else None
        # Only draw scores for predictions (Ground Truths usually lack scores or have dummy 0/-1 values)
        if score > 0.01:
            
            # Drawing boxes
            x = rec[0]
            y = rec[1]
            l = rec[3]
            w = rec[4]
            ang = rec[6]
            
            label = pred_labels[i]

            cls_name = ['Car','Pedestrian', 'Cyclist']
            color = color_dict[cls_name[label-1]] if label-1 < len(cls_name) else (128, 128, 128)
            rec = Rec((x, y), l, w, angle=ang, fill=False, color=color,lw=2)
            ax.add_patch(rec)
            # ------------------------------------------
            


            print(f"Drawing score {score:.2f} at ({x:.2f}, {y:.2f}) for frame")
            x = rec.get_x() + rec.get_width() if rec else 0   #top - right corner of the box
            y = rec.get_y() + rec.get_height() if rec else 0
            # Draw the text with a small black background so it's easy to read over point clouds
            ax.text(x, y, f"{score:.2f}", color='b', fontsize=7, 
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=0.5))
    # -----------------------------------

    legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]

    legend_elements += ext_legends
    
    ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)

def saveODImgs(anno, pts, img_path, color_dict, title='pred', fid = 0):
    print('=================== drawing images ===================')
    plt.rcParams['figure.dpi'] = 150
   
    ax = plt.gca()
    drawBEV(ax, pts, anno, color_dict, title)
    
    img_fname = P(f"{img_path}/{fid}.png")
    plt.xlim(0, 150)
    plt.ylim(-50, 50)
    plt.savefig(str(img_fname))
    plt.cla()
