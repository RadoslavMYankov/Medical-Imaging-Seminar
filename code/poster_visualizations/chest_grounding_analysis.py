import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from collections import defaultdict

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def parse_bbox(bbox_str):
    """Parse bounding box string to list of coordinates"""
    if isinstance(bbox_str, str):
        try:
            # Handle string representation of list
            if bbox_str.startswith('[') and bbox_str.endswith(']'):
                return ast.literal_eval(bbox_str)
            else:
                return []
        except:
            return []
    elif isinstance(bbox_str, list):
        return bbox_str
    else:
        return []

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

def load_llava_grounding_data(csv_path):
    """Load LLaVA-Med grounding results from CSV"""
    df = pd.read_csv(csv_path)
    
    # Parse bounding boxes
    df['ground_truth_parsed'] = df['ground_truth'].apply(parse_bbox)
    df['llava_prediction_parsed'] = df['prediction'].apply(parse_bbox)
    
    # Validate bounding boxes
    df['ground_truth_validated'] = df['ground_truth_parsed'].apply(lambda x: validate_and_clip_bbox(x))
    df['llava_prediction_validated'] = df['llava_prediction_parsed'].apply(lambda x: validate_and_clip_bbox(x))
    
    return df

def load_qwen_grounding_data(json_path):
    """Load Qwen grounding results from JSON"""
    with open(json_path, 'r') as f:
        qwen_data = json.load(f)
    
    qwen_df = pd.DataFrame(qwen_data)
    
    # Parse bounding boxes
    qwen_df['qwen_prediction_parsed'] = qwen_df['prediction'].apply(parse_bbox)
    qwen_df['qwen_prediction_validated'] = qwen_df['qwen_prediction_parsed'].apply(lambda x: validate_and_clip_bbox(x))
    
    return qwen_df

def merge_grounding_data(llava_df, qwen_df):
    """Merge LLaVA and Qwen grounding data"""
    # Create merge keys
    llava_df['merge_key'] = llava_df['img'].str.replace('.png', '') + '_' + llava_df['disease']
    qwen_df['merge_key'] = qwen_df['id'] + '_' + qwen_df['disease']
    
    # Merge dataframes
    merged_df = llava_df.merge(
        qwen_df[['merge_key', 'qwen_prediction_validated']], 
        on='merge_key', 
        how='inner'
    )
    
    return merged_df

def calculate_map_scores_per_image(merged_df):
    """Calculate mAP scores per image for both models"""
    image_results = []
    
    # Group by image
    for img_name in merged_df['img'].unique():
        img_data = merged_df[merged_df['img'] == img_name]
        
        # Collect valid bounding boxes
        gt_boxes = [box for box in img_data['ground_truth_validated'].tolist() if box is not None]
        llava_boxes = [box for box in img_data['llava_prediction_validated'].tolist() if box is not None]
        qwen_boxes = [box for box in img_data['qwen_prediction_validated'].tolist() if box is not None]
        
        # Calculate mAP scores using supervision library
        if gt_boxes:
            # LLaVA-Med scores
            if llava_boxes:
                gt_classes = [0] * len(gt_boxes)
                llava_classes = [0] * len(llava_boxes)
                llava_scores = compute_map_supervision(llava_boxes, llava_classes, gt_boxes, gt_classes)
            else:
                llava_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
            
            # Qwen scores
            if qwen_boxes:
                qwen_classes = [0] * len(qwen_boxes)
                qwen_scores = compute_map_supervision(qwen_boxes, qwen_classes, gt_boxes, gt_classes)
            else:
                qwen_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        else:
            llava_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
            qwen_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        
        image_results.append({
            'img': img_name,
            'num_gt_boxes': len(gt_boxes),
            'num_llava_boxes': len(llava_boxes),
            'num_qwen_boxes': len(qwen_boxes),
            'llava_map50_95': llava_scores['map50_95'],
            'llava_map50': llava_scores['map50'],
            'llava_map75': llava_scores['map75'],
            'qwen_map50_95': qwen_scores['map50_95'],
            'qwen_map50': qwen_scores['map50'],
            'qwen_map75': qwen_scores['map75']
        })
    
    return pd.DataFrame(image_results)

def calculate_map_scores_per_disease(merged_df):
    """Calculate mAP scores per disease for both models"""
    disease_results = []
    
    # Group by disease
    for disease in merged_df['disease'].unique():
        disease_data = merged_df[merged_df['disease'] == disease]
        
        # Collect all valid bounding boxes for this disease
        gt_boxes = [box for box in disease_data['ground_truth_validated'].tolist() if box is not None]
        llava_boxes = [box for box in disease_data['llava_prediction_validated'].tolist() if box is not None]
        qwen_boxes = [box for box in disease_data['qwen_prediction_validated'].tolist() if box is not None]
        
        # Calculate mAP scores using supervision library
        if gt_boxes:
            # LLaVA-Med scores
            if llava_boxes:
                gt_classes = [0] * len(gt_boxes)
                llava_classes = [0] * len(llava_boxes)
                llava_scores = compute_map_supervision(llava_boxes, llava_classes, gt_boxes, gt_classes)
            else:
                llava_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
            
            # Qwen scores
            if qwen_boxes:
                qwen_classes = [0] * len(qwen_boxes)
                qwen_scores = compute_map_supervision(qwen_boxes, qwen_classes, gt_boxes, gt_classes)
            else:
                qwen_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        else:
            llava_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
            qwen_scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        
        disease_results.append({
            'disease': disease,
            'num_gt_boxes': len(gt_boxes),
            'num_llava_boxes': len(llava_boxes),
            'num_qwen_boxes': len(qwen_boxes),
            'llava_map50_95': llava_scores['map50_95'],
            'llava_map50': llava_scores['map50'],
            'llava_map75': llava_scores['map75'],
            'qwen_map50_95': qwen_scores['map50_95'],
            'qwen_map50': qwen_scores['map50'],
            'qwen_map75': qwen_scores['map75']
        })
    
    return pd.DataFrame(disease_results)

def plot_map_scores_per_disease(disease_results_df):
    """Plot mAP scores comparison per disease"""
    # Prepare data for plotting
    diseases = disease_results_df['disease'].tolist()
    llava_map50 = disease_results_df['llava_map50'].tolist()
    qwen_map50 = disease_results_df['qwen_map50'].tolist()
    
    # Create DataFrame for seaborn
    plot_data = []
    for i, disease in enumerate(diseases):
        plot_data.append({'Disease': disease, 'Model': 'LLaVA-Med', 'mAP@50': llava_map50[i]})
        plot_data.append({'Disease': disease, 'Model': 'Qwen-2.5', 'mAP@50': qwen_map50[i]})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Define colors - light blue and light orange
    colors = [ '#FFB347', '#87CEEB']
    
    # Create grouped bar plot
    ax = sns.barplot(data=plot_df, x='Disease', y='mAP@50', hue='Model', 
                     palette=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('mAP@50 Score Comparison by Disease', fontsize=16, fontweight='bold')
    plt.xlabel('Disease', fontsize=12)
    plt.ylabel('mAP@50 Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Model', fontsize=10)
    
    # Add value labels on bars (only for non-zero values)
    for container in ax.containers:
        labels = []
        for v in container:
            height = v.get_height()
            if height > 0:
                labels.append(f'{height:.3f}')
            else:
                labels.append('')
        ax.bar_label(container, labels=labels, fontsize=8)
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()

def plot_overall_map_comparison(disease_results_df):
    """Plot overall mAP comparison"""
    # Calculate mean mAP scores
    llava_map50_mean = disease_results_df['llava_map50'].mean()
    llava_map75_mean = disease_results_df['llava_map75'].mean()
    llava_map50_95_mean = disease_results_df['llava_map50_95'].mean()
    
    qwen_map50_mean = disease_results_df['qwen_map50'].mean()
    qwen_map75_mean = disease_results_df['qwen_map75'].mean()
    qwen_map50_95_mean = disease_results_df['qwen_map50_95'].mean()
    
    # Create comparison plot
    metrics = ['mAP@50']
    llava_scores = [llava_map50_mean, llava_map75_mean, llava_map50_95_mean]
    qwen_scores = [qwen_map50_mean, qwen_map75_mean, qwen_map50_95_mean]
    
    # Prepare data for plotting
    plot_data = []
    for i, metric in enumerate(metrics):
        plot_data.append({'Metric': metric, 'Model': 'LLaVA-Med', 'Score': llava_scores[i]})
        plot_data.append({'Metric': metric, 'Model': 'Qwen-2.5', 'Score': qwen_scores[i]})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define colors - light blue and light orange
    colors = [ '#FFB347', '#87CEEB']
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model', 
                     palette=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('Overall mAP Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('mAP Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Model', fontsize=10)
    
    # Add value labels on bars (only for non-zero values)
    for container in ax.containers:
        labels = []
        for v in container:
            height = v.get_height()
            if height > 0:
                labels.append(f'{height:.3f}')
            else:
                labels.append('')
        ax.bar_label(container, labels=labels, fontsize=9)
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()

def create_qwen_grounding_results_csv(merged_df, output_path):
    """Create a CSV file with Qwen grounding results in the same format as LLaVA"""
    qwen_results = []
    
    for _, row in merged_df.iterrows():
        qwen_results.append({
            'img': row['img'],
            'disease': row['disease'],
            'ground_truth': row['ground_truth'],
            'prediction': str(row['qwen_prediction_validated']) if row['qwen_prediction_validated'] else "[]",
            'map50_95': 0.0,  # Will be calculated per image
            'map50': 0.0,
            'map75': 0.0
        })
    
    qwen_df = pd.DataFrame(qwen_results)
    
    # Calculate mAP scores per image for the CSV
    for img_name in qwen_df['img'].unique():
        img_data = merged_df[merged_df['img'] == img_name]
        
        gt_boxes = [box for box in img_data['ground_truth_validated'].tolist() if box is not None]
        qwen_boxes = [box for box in img_data['qwen_prediction_validated'].tolist() if box is not None]
        
        if gt_boxes and qwen_boxes:
            gt_classes = [0] * len(gt_boxes)
            qwen_classes = [0] * len(qwen_boxes)
            scores = compute_map_supervision(qwen_boxes, qwen_classes, gt_boxes, gt_classes)
        else:
            scores = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        
        # Update scores for this image
        mask = qwen_df['img'] == img_name
        qwen_df.loc[mask, 'map50_95'] = scores['map50_95']
        qwen_df.loc[mask, 'map50'] = scores['map50']
        qwen_df.loc[mask, 'map75'] = scores['map75']
    
    qwen_df.to_csv(output_path, index=False)
    print(f"Qwen grounding results saved to: {output_path}")
    
    return qwen_df

def main():
    """Main function to run the grounding analysis"""
    # Paths to your files
    llava_csv_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_grounding_bboxes.csv"
    qwen_json_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\qwen\\qwen2.5_grounding_results_per_disease.json"
    
    print("Loading grounding data...")
    
    try:
        # Load data
        llava_df = load_llava_grounding_data(llava_csv_path)
        qwen_df = load_qwen_grounding_data(qwen_json_path)
        
        print(f"Loaded LLaVA data with {len(llava_df)} samples")
        print(f"Loaded Qwen data with {len(qwen_df)} samples")
        
        # Merge data
        merged_df = merge_grounding_data(llava_df, qwen_df)
        print(f"Merged data contains {len(merged_df)} samples")
        
        # Display sample of merged data
        print("\nSample of merged data:")
        print(merged_df[['img', 'disease', 'ground_truth', 'prediction', 'qwen_prediction_validated']].head())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print("\nCalculating mAP scores per disease...")
    disease_results = calculate_map_scores_per_disease(merged_df)
    
    print("\n=== mAP Scores per Disease ===")
    print(disease_results.to_string(index=False))
    
    print("\nCalculating mAP scores per image...")
    image_results = calculate_map_scores_per_image(merged_df)
    
    print(f"\n=== Overall Statistics ===")
    print(f"LLaVA-Med - Mean mAP@50: {disease_results['llava_map50'].mean():.3f}")
    print(f"LLaVA-Med - Mean mAP@75: {disease_results['llava_map75'].mean():.3f}")
    print(f"LLaVA-Med - Mean mAP@50:95: {disease_results['llava_map50_95'].mean():.3f}")
    print(f"Qwen-2.5 - Mean mAP@50: {disease_results['qwen_map50'].mean():.3f}")
    print(f"Qwen-2.5 - Mean mAP@75: {disease_results['qwen_map75'].mean():.3f}")
    print(f"Qwen-2.5 - Mean mAP@50:95: {disease_results['qwen_map50_95'].mean():.3f}")
    
    print("\nPlotting mAP scores comparison per disease...")
    plot_map_scores_per_disease(disease_results)
    
    print("Plotting overall mAP comparison...")
    plot_overall_map_comparison(disease_results)
    
    # Save results
    print("\nSaving results...")
    qwen_output_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\qwen_grounding_bboxes.csv"
    qwen_results_df = create_qwen_grounding_results_csv(merged_df, qwen_output_path)

    disease_results_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\grounding_disease_comparison.csv"
    disease_results.to_csv(disease_results_path, index=False)
    print(f"Disease comparison results saved to: {disease_results_path}")

    image_results_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\grounding_image_comparison.csv"
    image_results.to_csv(image_results_path, index=False)
    print(f"Image comparison results saved to: {image_results_path}")


if __name__ == "__main__":
    main()