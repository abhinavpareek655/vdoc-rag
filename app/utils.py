def bbox_to_dict(bbox):
    x0, y0, x1, y1 = bbox
    return {'x0': int(x0), 'y0': int(y0), 'x1': int(x1), 'y1': int(y1)}
