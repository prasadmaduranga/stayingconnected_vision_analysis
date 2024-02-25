

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