import os
from os.path import isfile, join
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np

mypath = "/media/franco/hdd/dataset/nuscenes/samples/RADAR_FRONT_LEFT"
onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

rcs = []
means = []
stds = []
for fname in onlyfiles:
    file_path = join(mypath, fname)
    
    try:
        # This native class handles all 18 NuScenes radar channels automatically
        pc = RadarPointCloud.from_file(file_path)
        
        # pc.points is an (18, N) NumPy array where N is the number of points
        pc_data = pc.points.T[:,5]
        if len(pc_data) == 0:
            continue
        rcs.append(np.array(pc_data))
    
        means.append(np.mean(pc_data))
        stds.append(np.std(pc_data))
        
        
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"Extracted RCS values from {len(rcs)} radar point clouds.")
print(f"Mean RCS: {np.mean(means):.2f}, Std RCS: {np.mean(stds):.2f}")

bins = np.histogram(np.concatenate(rcs), range=(-10, 60), bins=140)[0]
print(f"Histogram bins: {bins.shape[0]}")
print(f"RCS histogram: {bins}")