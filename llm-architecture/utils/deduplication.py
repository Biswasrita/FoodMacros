import numpy as np

def calculate_overlap(box1, box2):
    """Calculates Intersection over Union (IoU) and Intersection over Smaller Box (IoS)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0, 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / float(union_area) if union_area > 0 else 0.0
    ios = inter_area / float(min(box1_area, box2_area))

    return iou, ios

def filter_redundant_crops(boxes, crops, iou_thresh=0.45, ios_thresh=0.70):
    """Filters out duplicate or highly overlapping bounding boxes to prevent double-counting calories."""
    if not boxes:
        return [], []

    # Sort boxes by confidence score (descending)
    sorted_indices = sorted(range(len(boxes)), key=lambda k: boxes[k]['conf'], reverse=True)
    keep_indices = []

    for i in sorted_indices:
        box_i = boxes[i]['box']
        redundant = False
        
        for j in keep_indices:
            box_j = boxes[j]['box']
            iou, ios = calculate_overlap(box_i, box_j)
            
            # If IoU > threshold OR box_i is > 70% inside an existing higher-confidence box
            if iou > iou_thresh or ios > ios_thresh:
                redundant = True
                break
                
        if not redundant:
            keep_indices.append(i)

    filtered_boxes = [boxes[k] for k in keep_indices]
    filtered_crops = [crops[k] for k in keep_indices]
    return filtered_boxes, filtered_crops