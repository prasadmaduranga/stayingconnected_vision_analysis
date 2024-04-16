
import numpy as np


def calculate_bbox(points):
    """
    Calculate the bounding box from a list of points.
    """
    x_coordinates, y_coordinates = zip(*points)
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)

def bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection Over Union (IOU) of two bounding boxes.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate IOU
    iou = inter_area / union_area if union_area else 0

    return iou

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at p2 formed by the line segments p1-p2 and p2-p3 using the law of cosines.
    """
    a = np.linalg.norm(p3 - p2)
    b = np.linalg.norm(p1 - p2)
    c = np.linalg.norm(p1 - p3)
    angle = np.arccos((b**2 + a**2 - c**2) / (2 * b * a))
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def calculate_object_center(x1, y1, x2, y2):
    """Calculate the center of a bounding box."""
    center_x = round((x1 + x2) / 2,4)
    center_y = round((y1 + y2) / 2,4)
    return center_x, center_y


def calculate_line_equation(p1, p2):
    """Calculate the slope (m) and y-intercept (b) of a line passing through points p1 and p2."""
    m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] - p1[0] != 0 else 0
    b = p1[1] - m * p1[0]
    return m, b

# Calculate the line equation for the shortest path
def calculate_line_equation(p1, p2):
    """Calculate the slope (m) and y-intercept (b) of a line passing through points p1 and p2."""
    m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] - p1[0] != 0 else 0
    b = p1[1] - m * p1[0]
    return m, b



