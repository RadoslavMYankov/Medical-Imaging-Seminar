import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pandas as pd
import re

def extract_bbox_from_text(text):
    """
    Extract bounding box coordinates from text descriptions.
    Looks for patterns like (x1, y1, x2, y2) in the text.
    """
    # Pattern to match coordinates in parentheses
    pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the first match and convert to floats
        x1, y1, x2, y2 = map(float, matches[0])
        return [x1, y1, x2, y2]
    return None

def load_annotations(annotations_file):
    """
    Load annotations from JSON file and extract bounding boxes by image name.
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    bbox_data = {}
    for case_id, case_data in annotations.items():
        if 'image_findings' in case_data and case_data['image_findings']:
            # Extract bounding boxes from image findings
            image_findings = case_data['image_findings']
            
            # Iterate through each image in the case
            for image_name, finding_data in image_findings.items():
                if isinstance(finding_data, dict) and 'bbox_2d_gold' in finding_data:
                    # Extract the actual bounding box coordinates from the nested array
                    bbox_list = finding_data['bbox_2d_gold']
                    if bbox_list and len(bbox_list) > 0:
                        # Flatten the nested structure - each bbox is [x1, y1, x2, y2]
                        image_bboxes = []
                        for bbox in bbox_list:
                            if isinstance(bbox, list) and len(bbox) == 4:
                                image_bboxes.append(bbox)
                        
                        if image_bboxes:
                            # Use the actual image name as the key
                            bbox_data[image_name] = image_bboxes
    
    return bbox_data

def load_predictions(predictions_file):
    """
    Load predictions from CSV file and extract bounding boxes.
    """
    df = pd.read_csv(predictions_file)
    predictions = {}
    
    for _, row in df.iterrows():
        image_name = row['image']
        prediction_text = row['prediction']
        
        bbox = extract_bbox_from_text(prediction_text)
        if bbox:
            predictions[image_name] = [bbox]  # Wrap in list for consistency
    
    return predictions

def boxes_to_detections(boxes, class_ids=None):
    """
    Convert list of [x1, y1, x2, y2] boxes into a sv.Detections object.
    
    Args:
        boxes: List of boxes [[x1,y1,x2,y2], ...]
        class_ids: Optional list/array of class IDs for each box. 
                   If None, defaults to class_id=0 for all boxes.
    """
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    xyxy = np.array(boxes, dtype=np.float32)
    
    if class_ids is None:
        class_id_arr = np.zeros(len(boxes), dtype=int)
    else:
        class_id_arr = np.array(class_ids, dtype=int)
        if len(class_id_arr) != len(boxes):
            raise ValueError("Length of class_ids must match number of boxes")
    
    confidence = np.ones(len(boxes), dtype=np.float32)  # fixed confidence = 1.0
    
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id_arr,
        confidence=confidence
    )

def compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes):
    """
    Compute mAP scores using supervision library.
    """
    if not pred_boxes or not true_boxes:
        return None
    
    preds = boxes_to_detections(pred_boxes, pred_classes)
    targets = boxes_to_detections(true_boxes, true_classes)
    
    metric = MeanAveragePrecision()
    result = metric.update(preds, targets).compute()
    
    return {
        'map50_95': result.map50_95,
        'map50': result.map50,
        'map75': result.map75
    }

def create_evaluation_csv(predictions_file, annotations_file, output_file):
    """
    Create CSV file with image, ground truth, prediction, and mAP scores.
    """
    # Load data
    predictions = load_predictions(predictions_file)
    annotations = load_annotations(annotations_file)
    
    results = []
    
    for image_name, pred_boxes in predictions.items():
        # Get ground truth boxes for this specific image
        true_boxes = annotations.get(image_name, [])
        
        # Compute mAP scores
        if pred_boxes and true_boxes:
            pred_classes = [0] * len(pred_boxes)  # Assuming single class
            true_classes = [0] * len(true_boxes)  # Assuming single class
            
            map_scores = compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes)
        else:
            map_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        
        results.append({
            'image': image_name,
            'ground_truth': str(true_boxes) if true_boxes else "[]",
            'prediction': str(pred_boxes) if pred_boxes else "[]",
            'map50_95': map_scores['map50_95'] if map_scores else 0.0,
            'map50': map_scores['map50'] if map_scores else 0.0,
            'map75': map_scores['map75'] if map_scores else 0.0
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return df

if __name__ == "__main__":
    # File paths
    predictions_file = "/home/useradd/seminar/Mecial-Imaging-Seminar/results/mri_predictions_grounding.csv"
    annotations_file = "/home/useradd/seminar/Mecial-Imaging-Seminar/src/data/nova_brain/annotations.json"
    output_file = "/home/useradd/seminar/Mecial-Imaging-Seminar/results/mri_grounding_bboxes.csv"
    
    # Create evaluation CSV
    results_df = create_evaluation_csv(predictions_file, annotations_file, output_file)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Total images processed: {len(results_df)}")
    print(f"Average mAP@50: {results_df['map50'].mean():.4f}")
    print(f"Average mAP@75: {results_df['map75'].mean():.4f}")
    print(f"Average mAP@50:95: {results_df['map50_95'].mean():.4f}")
    
    # Show first few results
    print("\nFirst 5 results:")
    print(results_df.head())