def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1) * abs(y2 - y1)
