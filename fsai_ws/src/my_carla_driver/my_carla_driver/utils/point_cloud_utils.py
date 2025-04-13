import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

def create_points_field() -> list:
    fields = []
    fields.append(PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1))
    fields.append(PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1))
    fields.append(PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1))
    fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
    return fields

def create_point_cloud2(points: np.ndarray, frame_id: str) -> PointCloud2:
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    
    if len(points.shape) == 2:
        msg.height = 1
        msg.width = points.shape[0]
    else:
        msg.height = points.shape[0]
        msg.width = points.shape[1]

    msg.fields = create_points_field()
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    
    return msg 