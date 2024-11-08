import pandas as pd

def calculate_iou(box_a, box_b):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes. """
    xa1, ya1, wa, ha = box_a
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = box_b
    xb2, yb2 = xb1 + wb, yb1 + hb

    inter_width = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_height = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_width * inter_height

    box_a_area = wa * ha
    box_b_area = wb * hb
    union_area = box_a_area + box_b_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def evaluate_metrics(detected_objects, ground_truth, correct_classifications):
    """
    Parameters:
    - detected_objects: list of tuples (image_name, bbox, predicted_class)
    - ground_truth: list of tuples (image_name, bbox, true_class)
    - correct_classifications: dictionary mapping image_name to correct class label

    Returns:
    - total_score: int, the final computed metric score
    """
    iou_threshold = 0.5
    classification_points = {'correct': 5, 'wrong': -5}
    detection_points = {'correct': 1, 'missed': -1, 'false_positive': -1}

    total_score = 0
    used_ground_truths = set()

    # Calculate detection score
    for detected in detected_objects:
        print(detected)
        image_name, detected_bbox, detected_class = detected
        best_iou = 0
        best_gt = None

        for gt in ground_truth:
            gt_image_name, gt_bbox, true_class = gt
            if gt_image_name == image_name and gt not in used_ground_truths:
                iou = calculate_iou(detected_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        if best_iou > iou_threshold:
            used_ground_truths.add(best_gt)
            total_score += detection_points['correct']

            # Classification score
            _, _, true_class = best_gt
            if detected_class == true_class:
                total_score += classification_points['correct']
            else:
                total_score += classification_points['wrong']
        else:
            total_score += detection_points['false_positive']

    for image_name, _, _ in ground_truth:
        if image_name not in (x[0] for x in detected_objects):
            total_score += detection_points['missed']

    return total_score


# Example usage:
detected_objects = list(pd.read_csv('submissions/submission.csv').itertuples(index=False, name=None))
print(detected_objects)

ground_truth = list(pd.read_csv('submissions/annotation.csv').itertuples(index=False, name=None))

score = evaluate_metrics(detected_objects, ground_truth, {})
print("Total Score:", score)
