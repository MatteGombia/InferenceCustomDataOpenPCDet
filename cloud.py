import numpy as np

PointXYZ = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
])
PointXYZI = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
    ("intensity", np.float32),
    ("st_1", "V12"),
])
PointXYZICT = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
    ("intensity", np.float32),
    ("channel", np.uint16),
    ("st_1", np.uint16),
    ("time", np.float32),
    ("st_2", "V4"),
])
PointXYZL = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
    ("label", np.uint32),
    ("st_1", "V12"),
])
PointXYZTMRV = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("time", np.float32),
    ("moving", np.uint8),
    ("st_0", np.uint8),
    ("st_1", np.uint16),
    ("range_var", np.float32),
    ("vel_var", np.float32),
    ("st_2", "V4"),
])
PointXYZIVT = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
    ("intensity", np.float32),
    ("vel_var", np.float32),
    ("time", np.float32),
    ("st_1", "V4"),
])
PointXYZICTRGB = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
    ("intensity", np.float32),
    ("channel", np.uint16),
    ("st_1", np.uint16),
    ("time", np.float32),
    ("r", np.float32),
    ("g", np.float32),
    ("b", np.float32),
    ("st_2", "V8"),
])
PointRadarInfo = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("st_0", "V4"),
    ("intensity", np.float32),
    ("snr", np.float32),
    ("max_unambiguous_velocity", np.float32),
    ("radial_ambiguous_velocity", np.float32),
    ("radial_unambiguous_velocity", np.float32),
    ("compensated_vx", np.float32),
    ("compensated_vy", np.float32),
    ("time", np.float32),
])

POINT_IDX = {
    0: PointXYZ,
    1: PointXYZI,
    2: PointXYZICT,
    3: PointXYZL,
    4: PointXYZTMRV,
    5: PointXYZIVT,
    6: PointXYZICTRGB,
    7: PointRadarInfo,
}

POINT_TYPE = {
    PointXYZ: "PointXYZ",
    PointXYZI: "PointXYZI",
    PointXYZICT: "PointXYZICT",
    PointXYZL: "PointXYZL",
    PointXYZTMRV: "PointXYZTMRV",
    PointXYZIVT: "PointXYZIVT",
    PointXYZICTRGB: "PointXYZICTRGB",
    PointRadarInfo: "PointRadarInfo",
}

def protoCloudToNumpy(msg):
    data = np.frombuffer(msg.data, dtype=POINT_IDX[msg.type], count=msg.height)
    fields = [name for name in data.dtype.names if 'st_' not in name]
    return np.array([[float(x) for x in tup] for tup in data[fields]])

def typeToPoint(typept):
    return POINT_TYPE[POINT_IDX[typept]]