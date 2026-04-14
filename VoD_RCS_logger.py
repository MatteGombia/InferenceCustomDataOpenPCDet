from os import listdir
from os.path import isfile, join
import numpy as np

mypath = "/media/franco/hdd/dataset/VoD/view_of_delft_PUBLIC/radar/training/velodyne/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(f"Found {len(onlyfiles)} radar point cloud files in {mypath}. Processing SNR statistics...")


std = []
bins = []
mean = []
for fname in onlyfiles:
    try:
        radar_point_cloud = np.fromfile(str(join(mypath, fname)), dtype=np.float32).reshape(-1, 7)
        
        std.append(np.std(radar_point_cloud[:, 3]))
        mean.append(np.mean(radar_point_cloud[:, 3]))
        bins.append(np.histogram(radar_point_cloud[:, 3], range=(-60, 40), bins=1000)[0])
    except Exception as e:
        print(f"Error processing file {fname}: {e}")
    #print(f"File: {fname}, SNR mean: {mean[-1]:.2f}, SNR std: {std[-1]:.2f}")
print(f"Overall SNR mean: {np.mean(mean):.2f}, Overall SNR std: {np.mean(std):.2f}")
print(f"Overall SNR histogram (sum of all files): {np.sum(bins, axis=0)}")

# # Overall SNR mean: -12.43, Overall SNR std: 13.27

