#TRUCKSCENES
import pypcd4
import numpy as np
from os import listdir
from os.path import isfile, join


mypath = "/media/franco/hdd/dataset/man-truckscenes/man-truckscenes/v1.1-trainval/sweeps/RADAR_LEFT_FRONT"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
std = []
mean = []
bins = []
for fname in onlyfiles:
    try:
        radar = pypcd4.PointCloud.from_path(str(join(mypath, fname)))
        std.append(np.std(radar.pc_data["rcs"]))
        mean.append(np.mean(radar.pc_data["rcs"]))
        bins.append(np.histogram(radar.pc_data["rcs"], range=(-40, 60), bins=100)[0])
    except Exception as e:
        print(f"Error processing file {fname}: {e}")
    
print("Overall RCS mean: {:.2f}, Overall RCS std: {:.2f}".format(np.mean(mean), np.mean(std)))
print("Overall RCS histogram (sum of all files): {}".format(np.sum(bins, axis=0)))

#Overall RCS mean: -6.70, Overall RCS std: 8.43

