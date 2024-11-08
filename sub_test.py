import numpy as np
import pandas as pd
import torch
import torchvision.ops.boxes as bops


def calculate_iou_vectorized(detected_bboxes, gt_bboxes):
    # Extract coordinates
    xa1, ya1, wa, ha = detected_bboxes[:, 0], detected_bboxes[:, 1], detected_bboxes[:, 2], detected_bboxes[:, 3]
    xb1, yb1, wb, hb = gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2], gt_bboxes[:, 3]

    # Convert center (xc, yc) to (x1, y1)
    xa1, ya1 = xa1 - wa / 2, ya1 - ha / 2
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1 = xb1 - wb / 2, yb1 - hb / 2
    xb2, yb2 = xb1 + wb, yb1 + hb

    # Calculate intersection
    inter_width = np.maximum(0, np.minimum(xa2, xb2) - np.maximum(xa1, xb1))
    inter_height = np.maximum(0, np.minimum(ya2, yb2) - np.maximum(ya1, yb1))
    inter_area = inter_width * inter_height

    # Calculate union
    box_a_area = wa * ha
    box_b_area = wb * hb
    union_area = box_a_area + box_b_area - inter_area

    iou = inter_area / union_area
    return iou


def evaluate_metrics(detected_objects, ground_truth):
    iou_threshold = 0.5
    classification_points = {'correct': 5, 'wrong': -5}
    detection_points = {'correct': 1, 'missed': -1, 'false_positive': -1}

    # Merge detected and ground truth data on image name
    merged_df = pd.merge(detected_objects, ground_truth, on="Name", suffixes=('_det', '_gt'))

    # Convert bbox strings to float arrays
    detected_bboxes = merged_df['Bbox_det'].apply(lambda x: list(map(float, x.split(',')))).tolist()
    gt_bboxes = merged_df['Bbox_gt'].apply(lambda x: list(map(float, x.split(',')))).tolist()

    detected_bboxes = np.array(detected_bboxes)
    gt_bboxes = np.array(gt_bboxes)

    # Calculate IoU for each pair of detected and ground truth bboxes
    ious = calculate_iou_vectorized(detected_bboxes, gt_bboxes)

    # Determine correct detections based on IoU threshold
    correct_detections_mask = ious > iou_threshold
    correct_detections = merged_df[correct_detections_mask]

    # Calculate scores for correct detections
    total_score = (correct_detections['Class_det'] == correct_detections['Class_gt']).sum() * classification_points[
        'correct']
    total_score += (correct_detections['Class_det'] != correct_detections['Class_gt']).sum() * classification_points[
        'wrong']

    # Add detection points for correct detections
    total_score += correct_detections_mask.sum() * detection_points['correct']

    # Calculate false positives
    false_positives = (~correct_detections_mask).sum()
    total_score += false_positives * detection_points['false_positive']

    # Calculate missed detections
    all_detected_images = set(detected_objects["Name"])
    all_ground_truth_images = set(ground_truth["Name"])

    missed_images = all_ground_truth_images - all_detected_images
    total_score += len(missed_images) * detection_points['missed']
    n = len(ground_truth)
    print(f"Raw score: {total_score}  Max raw score {n * 6}")
    return total_score / (n * 6)

# Example usage:
names = ["Name", "Bbox","Class"]
detected_objects = pd.read_csv('submissions/submission.csv',names= names, header=0)
print(detected_objects)

ground_truth = pd.read_csv('submissions/annotation.csv',names= names,header=0)

score = evaluate_metrics(detected_objects, ground_truth)
print("Total Score:", score)
