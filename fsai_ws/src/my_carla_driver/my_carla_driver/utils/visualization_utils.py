from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

def create_cone_marker(position: Point, color: ColorRGBA, id: int, frame_id: str) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.id = id
    marker.type = Marker.CONE
    marker.action = Marker.ADD
    marker.pose.position = position
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.3  # base diameter
    marker.scale.y = 0.3  # base diameter
    marker.scale.z = 0.5  # height
    marker.color = color
    return marker

def create_path_marker(points: list, color: ColorRGBA, id: int, frame_id: str) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.id = id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1  # line width
    marker.points = points
    marker.color = color
    return marker 