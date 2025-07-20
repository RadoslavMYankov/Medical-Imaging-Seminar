import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pandas as pd
import re

def validate_and_clip_bbox(bbox, image_width=None, image_height=None):
    """
    Validate and clip bounding box coordinates to image bounds.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        image_width: Width of the image (optional)
        image_height: Height of the image (optional)
    
    Returns:
        Valid bounding box clipped to image bounds, or None if invalid
    """
    if not bbox or len(bbox) != 4:
        return None
    
    x1, y1, x2, y2 = bbox
    
    # Ensure x1 <= x2 and y1 <= y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Clip to image bounds if provided
    if image_width is not None:
        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
    
    if image_height is not None:
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))
    
    # Check if box still has valid area after clipping
    if x2 <= x1 or y2 <= y1:
        return None
    
    return [x1, y1, x2, y2]

def extract_bbox_from_text(text, image_width=None, image_height=None):
    """
    Extract bounding box coordinates from text descriptions.
    Looks for patterns like (x1, y1, x2, y2) in the text.
    
    Args:
        text: Text containing bounding box coordinates
        image_width: Width of the image for validation (optional)
        image_height: Height of the image for validation (optional)
    """
    # Pattern to match coordinates in parentheses
    pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the first match and convert to floats
        x1, y1, x2, y2 = map(float, matches[0])
        bbox = [x1, y1, x2, y2]
        # Validate and clip the bounding box
        return validate_and_clip_bbox(bbox, image_width, image_height)
    return None

def extract_disease_from_text(text):
    """
    Extract disease name from prediction text.
    """
    # Look for common patterns like "The [Disease] is located"
    patterns = [
        r'The\s+([^.]+?)\s+is\s+located',
        r'bounding\s+box\s+coordinates\s+for\s+the\s+([^.]+?)\s+(?:in|are)',
        r'coordinates\s+for\s+the\s+([^.]+?)\s+'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            disease = match.group(1).strip()
            # Clean up common artifacts
            disease = re.sub(r'\s+in\s+the.*', '', disease, flags=re.IGNORECASE)
            disease = re.sub(r'\s+chest.*', '', disease, flags=re.IGNORECASE)
            return disease
    
    return "Unknown"

def load_annotations(annotations_file, image_width=None, image_height=None):
    """
    Load annotations from JSON file and extract individual bounding boxes with diseases.
    Returns a list where each entry is one bounding box with its disease.
    
    Args:
        annotations_file: Path to annotations JSON file
        image_width: Width of images for validation (optional)
        image_height: Height of images for validation (optional)
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    annotation_rows = []
    
    for case_id, case_data in annotations.items():
        if 'bbox_2d' in case_data and case_data['bbox_2d']:
            # Extract individual bounding boxes from bbox_2d
            for bbox_entry in case_data['bbox_2d']:
                if len(bbox_entry) >= 5:  # [x1, y1, x2, y2, disease]
                    x1, y1, x2, y2, disease = bbox_entry[:5]
                    bbox = [x1, y1, x2, y2]
                    # Validate and clip the bounding box
                    validated_bbox = validate_and_clip_bbox(bbox, image_width, image_height)
                    if validated_bbox:
                        annotation_rows.append({
                            'case_id': case_id,
                            'bbox': validated_bbox,
                            'disease': disease
                        })
    
    return annotation_rows

def load_predictions(predictions_file, image_width=None, image_height=None):
    """
    Load predictions from CSV file and extract bounding boxes with diseases.
    
    Args:
        predictions_file: Path to predictions CSV file
        image_width: Width of images for validation (optional)
        image_height: Height of images for validation (optional)
    """
    df = pd.read_csv(predictions_file)
    predictions = []
    
    for _, row in df.iterrows():
        image_name = row['image']
        prediction_text = row['prediction']
        
        bbox = extract_bbox_from_text(prediction_text, image_width, image_height)
        disease = extract_disease_from_text(prediction_text)
        
        if bbox:
            predictions.append({
                'image': image_name,
                'bbox': bbox,
                'disease': disease,
                'prediction_text': prediction_text
            })
    
    return predictions

def boxes_to_detections(boxes, class_ids=None):
    """
    Convert list of [x1, y1, x2, y2] boxes into a sv.Detections object.
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
    
    confidence = np.ones(len(boxes), dtype=np.float32)
    
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
        return {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
    
    preds = boxes_to_detections(pred_boxes, pred_classes)
    targets = boxes_to_detections(true_boxes, true_classes)
    
    metric = MeanAveragePrecision()
    result = metric.update(preds, targets).compute()
    
    return {
        'map50_95': result.map50_95,
        'map50': result.map50,
        'map75': result.map75
    }

def create_xray_grounding_bboxes_csv(predictions_file, annotations_file, output_file, image_width=None, image_height=None):
    """
    Create CSV file with columns: img, disease, ground_truth, prediction, map_scores
    One line per bounding box from annotations.
    
    Args:
        predictions_file: Path to predictions CSV file
        annotations_file: Path to annotations JSON file
        output_file: Path to output CSV file
        image_width: Width of images for bounding box validation (optional)
        image_height: Height of images for bounding box validation (optional)
    """
    # Load data
    predictions = load_predictions(predictions_file, image_width, image_height)
    annotation_rows = load_annotations(annotations_file, image_width, image_height)
    
    results = []
    
    # Process each annotation bounding box
    for ann_row in annotation_rows:
        case_id = ann_row['case_id']
        ann_bbox = ann_row['bbox']
        ann_disease = ann_row['disease']
        
        # Find predictions for this case (assuming image name matches case_id + .png)
        image_name = f"{case_id}.png"
        case_predictions = [p for p in predictions if p['image'] == image_name]
        
        # Get all prediction bboxes for this image for mAP calculation
        pred_bboxes = [p['bbox'] for p in case_predictions]
        
        # Get all ground truth bboxes for this case for mAP calculation
        case_gt_bboxes = [row['bbox'] for row in annotation_rows if row['case_id'] == case_id]
        
        # Compute mAP for this case
        if pred_bboxes and case_gt_bboxes:
            pred_classes = [0] * len(pred_bboxes)  # Single class for now
            gt_classes = [0] * len(case_gt_bboxes)
            map_scores = compute_map_supervision(pred_bboxes, pred_classes, case_gt_bboxes, gt_classes)
        else:
            map_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        
        # Find the best matching prediction for this specific annotation
        best_pred = None
        if case_predictions:
            # For now, just take the first prediction with the same disease, or first prediction
            disease_matches = [p for p in case_predictions if p['disease'].lower() == ann_disease.lower()]
            if disease_matches:
                best_pred = disease_matches[0]
            else:
                best_pred = case_predictions[0]
        
        # Create row for this annotation bounding box
        results.append({
            'img': image_name,
            'disease': ann_disease,
            'ground_truth': str(ann_bbox),
            'prediction': str(best_pred['bbox']) if best_pred else "[]",
            'map50_95': map_scores['map50_95'],
            'map50': map_scores['map50'],
            'map75': map_scores['map75']
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Total bounding boxes processed: {len(df)}")
    return df

if __name__ == "__main__":
    # File paths
    predictions_file = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_predictions_grounding.csv"
    annotations_file = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\src\\data\\chest_xrays\\annotations_len_50.json"
    output_file = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_grounding_bboxes.csv"

    # Optional: Set image dimensions for bounding box validation
    # If you know the image dimensions, uncomment and set these values:
    # IMAGE_WIDTH = 1024   # Replace with actual image width
    # IMAGE_HEIGHT = 1024  # Replace with actual image height
    IMAGE_WIDTH = None
    IMAGE_HEIGHT = None
    
    # Create evaluation CSV
    results_df = create_xray_grounding_bboxes_csv(predictions_file, annotations_file, output_file,
                                                IMAGE_WIDTH, IMAGE_HEIGHT)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Total bounding boxes processed: {len(results_df)}")
    print(f"Unique images: {results_df['img'].nunique()}")
    print(f"Unique diseases: {results_df['disease'].nunique()}")
    print(f"Average mAP@50: {results_df['map50'].mean():.4f}")
    print(f"Average mAP@75: {results_df['map75'].mean():.4f}")
    print(f"Average mAP@50:95: {results_df['map50_95'].mean():.4f}")
    
    # Show disease distribution
    print("\nDisease distribution:")
    print(results_df['disease'].value_counts())
    
    # Show first few results
    print("\nFirst 5 results:")
    print(results_df.head())