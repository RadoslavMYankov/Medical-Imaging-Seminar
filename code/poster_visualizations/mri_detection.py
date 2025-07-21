import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from collections import defaultdict
import ast
import re

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def parse_bbox(bbox_str):
    """Parse bounding box string to list of coordinates"""
    if isinstance(bbox_str, str):
        try:
            # Handle different string formats
            if bbox_str.strip() == '[]' or bbox_str.strip() == '':
                return []
            
            # Try to evaluate as Python literal
            if bbox_str.startswith('[[') and bbox_str.endswith(']]'):
                return ast.literal_eval(bbox_str)
            elif bbox_str.startswith('[') and bbox_str.endswith(']'):
                # Single bbox or list of coordinates
                coords = ast.literal_eval(bbox_str)
                if len(coords) == 4:  # Single bbox
                    return [coords]
                return coords
            else:
                # Try to extract numbers with regex
                numbers = re.findall(r'-?\d+\.?\d*', bbox_str)
                if len(numbers) >= 4:
                    coords = [float(x) for x in numbers[:4]]
                    return [coords]
                return []
        except:
            return []
    elif isinstance(bbox_str, list):
        return bbox_str
    else:
        return []

def parse_qwen_prediction(pred_str):
    """Parse Qwen prediction string that might contain multiple bboxes"""
    if isinstance(pred_str, str):
        try:
            if pred_str.strip() == '[]' or pred_str.strip() == '':
                return []
            
            # Handle multiple bboxes like "[350,428,460,555] [585,430,679,555]"
            if ']' in pred_str and '[' in pred_str:
                # Split by '] [' to get individual boxes
                bbox_parts = re.findall(r'\[([^\]]+)\]', pred_str)
                bboxes = []
                for part in bbox_parts:
                    coords = [float(x.strip()) for x in part.split(',')]
                    if len(coords) == 4:
                        bboxes.append(coords)
                return bboxes
            
            # Single bbox
            coords = ast.literal_eval(pred_str)
            if len(coords) == 4:
                return [coords]
            return coords
        except:
            return []
    return []

def validate_and_clip_bbox(bbox, image_width=1024, image_height=1024):
    """
    Validate and clip bounding box coordinates to image bounds.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        image_width: Width of the image (default 1024)
        image_height: Height of the image (default 1024)
    
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
    
    # Clip to image bounds
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
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
            class_id_arr = np.zeros(len(boxes), dtype=int)
    
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

def load_llava_mri_grounding_data(csv_path):
    """Load LLaVA-Med MRI grounding results from CSV"""
    df = pd.read_csv(csv_path)
    
    # Parse bounding boxes
    df['ground_truth_parsed'] = df['ground_truth'].apply(parse_bbox)
    df['llava_prediction_parsed'] = df['prediction'].apply(parse_bbox)
    
    # Validate bounding boxes - flatten lists and validate each box
    df['ground_truth_validated'] = df['ground_truth_parsed'].apply(
        lambda boxes: [validate_and_clip_bbox(box) for box in boxes if validate_and_clip_bbox(box) is not None]
    )
    df['llava_prediction_validated'] = df['llava_prediction_parsed'].apply(
        lambda boxes: [validate_and_clip_bbox(box) for box in boxes if validate_and_clip_bbox(box) is not None]
    )
    
    return df

def load_qwen_mri_detection_data(json_path):
    """Load Qwen MRI detection results from JSON"""
    with open(json_path, 'r') as f:
        qwen_data = json.load(f)
    
    qwen_df = pd.DataFrame(qwen_data)
    
    # Parse bounding boxes
    qwen_df['qwen_prediction_parsed'] = qwen_df['prediction'].apply(parse_qwen_prediction)
    qwen_df['qwen_prediction_validated'] = qwen_df['qwen_prediction_parsed'].apply(
        lambda boxes: [validate_and_clip_bbox(box) for box in boxes if validate_and_clip_bbox(box) is not None]
    )
    
    return qwen_df

def merge_mri_grounding_data(llava_df, qwen_df):
    """Merge LLaVA and Qwen MRI grounding data"""
    # Merge on image name
    merged_df = llava_df.merge(
        qwen_df[['image', 'qwen_prediction_validated']], 
        on='image', 
        how='inner'
    )
    
    return merged_df

def calculate_map_scores_per_image(merged_df):
    """Calculate mAP scores per image for both models"""
    image_results = []
    
    # Process each image
    for _, row in merged_df.iterrows():
        img_name = row['image']
        gt_boxes = row['ground_truth_validated']
        llava_boxes = row['llava_prediction_validated']
        qwen_boxes = row['qwen_prediction_validated']
        
        # Calculate mAP scores using supervision library
        if gt_boxes:
            # LLaVA-Med scores
            llava_map = compute_map_supervision(
                llava_boxes, 
                [0] * len(llava_boxes) if llava_boxes else [],
                gt_boxes, 
                [0] * len(gt_boxes)
            )
            
            # Qwen scores  
            qwen_map = compute_map_supervision(
                qwen_boxes, 
                [0] * len(qwen_boxes) if qwen_boxes else [],
                gt_boxes, 
                [0] * len(gt_boxes)
            )
        else:
            llava_map = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
            qwen_map = {'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0}
        
        image_results.append({
            'image': img_name,
            'llava_map50': llava_map['map50'],
            'llava_map75': llava_map['map75'],
            'llava_map50_95': llava_map['map50_95'],
            'qwen_map50': qwen_map['map50'],
            'qwen_map75': qwen_map['map75'],
            'qwen_map50_95': qwen_map['map50_95'],
            'gt_boxes_count': len(gt_boxes),
            'llava_boxes_count': len(llava_boxes),
            'qwen_boxes_count': len(qwen_boxes)
        })
    
    return pd.DataFrame(image_results)

def plot_overall_map_comparison(image_results_df):
    """Plot overall mAP comparison"""
    # Calculate mean mAP scores
    llava_map50_mean = image_results_df['llava_map50'].mean()
    llava_map75_mean = image_results_df['llava_map75'].mean()
    llava_map50_95_mean = image_results_df['llava_map50_95'].mean()
    
    qwen_map50_mean = image_results_df['qwen_map50'].mean()
    qwen_map75_mean = image_results_df['qwen_map75'].mean()
    qwen_map50_95_mean = image_results_df['qwen_map50_95'].mean()
    
    # Create comparison plot
    metrics = ['mAP@50', 'mAP@75', 'mAP@50:95']
    llava_scores = [llava_map50_mean, llava_map75_mean, llava_map50_95_mean]
    qwen_scores = [qwen_map50_mean, qwen_map75_mean, qwen_map50_95_mean]
    
    # Prepare data for plotting
    plot_data = []
    for i, metric in enumerate(metrics):
        plot_data.append({'Metric': metric, 'Model': 'LLaVA-Med', 'Score': llava_scores[i]})
        plot_data.append({'Metric': metric, 'Model': 'Qwen-2.5', 'Score': qwen_scores[i]})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define colors - orange for LLaVA-Med, light blue for Qwen-2.5
    colors = ['#FFB347', '#87CEEB']
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model', 
                     palette=colors, alpha=0.8, edgecolor='black', linewidth=1,
                     hue_order=['LLaVA-Med', 'Qwen-2.5'])

    plt.title('Overall mAP Score for MRI Detection', fontsize=16, fontweight='bold')
    plt.ylabel('mAP Score', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Model', fontsize=14)
    
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

def create_summary_statistics(image_results_df):
    """Create summary statistics table"""
    summary_stats = {
        'Metric': ['mAP@50', 'mAP@75', 'mAP@50:95'],
        'LLaVA-Med Mean': [
            image_results_df['llava_map50'].mean(),
            image_results_df['llava_map75'].mean(),
            image_results_df['llava_map50_95'].mean()
        ],
        'LLaVA-Med Std': [
            image_results_df['llava_map50'].std(),
            image_results_df['llava_map75'].std(),
            image_results_df['llava_map50_95'].std()
        ],
        'Qwen-2.5 Mean': [
            image_results_df['qwen_map50'].mean(),
            image_results_df['qwen_map75'].mean(),
            image_results_df['qwen_map50_95'].mean()
        ],
        'Qwen-2.5 Std': [
            image_results_df['qwen_map50'].std(),
            image_results_df['qwen_map75'].std(),
            image_results_df['qwen_map50_95'].std()
        ]
    }
    
    # Add which model is better
    summary_stats['Better Model'] = []
    for i in range(3):
        if summary_stats['LLaVA-Med Mean'][i] > summary_stats['Qwen-2.5 Mean'][i]:
            summary_stats['Better Model'].append('LLaVA-Med')
        else:
            summary_stats['Better Model'].append('Qwen-2.5')
    
    return pd.DataFrame(summary_stats)

def save_results(summary_df, image_results_df, base_path):
    """Save all results to files"""

    
    # Save summary statistics
    summary_path = f"{base_path}_summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = f"{base_path}_detailed_results.csv"
    image_results_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")

def main():
    """Main function to run the MRI detection analysis"""
    # Paths to your files
    llava_csv_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_grounding_bboxes.csv"
    qwen_json_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\qwen\\qwen2_5_detection_results.json"
    
    print("Loading MRI detection data...")
    
    try:
        # Load data
        llava_df = load_llava_mri_grounding_data(llava_csv_path)
        qwen_df = load_qwen_mri_detection_data(qwen_json_path)
        
        print(f"Loaded LLaVA-Med data for {len(llava_df)} images")
        print(f"Loaded Qwen-2.5 data for {len(qwen_df)} images")
        
        # Merge data
        merged_df = merge_mri_grounding_data(llava_df, qwen_df)
        print(f"Merged data contains {len(merged_df)} images")
        
        # Display sample of merged data
        print("\nSample of merged data:")
        print(merged_df[['image', 'ground_truth_validated', 'llava_prediction_validated', 'qwen_prediction_validated']].head())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print("\nCalculating mAP scores per image...")
    image_results = calculate_map_scores_per_image(merged_df)
    
    # Create summary statistics
    summary_df = create_summary_statistics(image_results)
    
    print("\n=== MRI Detection mAP Statistics ===")
    print(summary_df.to_string(index=False))
    
    print(f"\n=== Overall Statistics ===")
    print(f"LLaVA-Med - Mean mAP@50: {image_results['llava_map50'].mean():.4f} ± {image_results['llava_map50'].std():.4f}")
    print(f"LLaVA-Med - Mean mAP@75: {image_results['llava_map75'].mean():.4f} ± {image_results['llava_map75'].std():.4f}")
    print(f"LLaVA-Med - Mean mAP@50:95: {image_results['llava_map50_95'].mean():.4f} ± {image_results['llava_map50_95'].std():.4f}")
    print(f"Qwen-2.5 - Mean mAP@50: {image_results['qwen_map50'].mean():.4f} ± {image_results['qwen_map50'].std():.4f}")
    print(f"Qwen-2.5 - Mean mAP@75: {image_results['qwen_map75'].mean():.4f} ± {image_results['qwen_map75'].std():.4f}")
    print(f"Qwen-2.5 - Mean mAP@50:95: {image_results['qwen_map50_95'].mean():.4f} ± {image_results['qwen_map50_95'].std():.4f}")

    print("\nPlotting overall mAP comparison...")
    plot_overall_map_comparison(image_results)
    
    # Save results
    print("\nSaving results...")
    base_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_detection_comparison"
    save_results(summary_df, image_results, base_path)

if __name__ == "__main__":
    main()
