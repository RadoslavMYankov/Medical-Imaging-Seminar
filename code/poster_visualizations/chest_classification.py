import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_csv_data(csv_path):
    """Load the CSV data with original predictions and labels"""
    df = pd.read_csv(csv_path)
    return df

def load_qwen_predictions(json_path):
    """Load Qwen predictions from JSON file"""
    with open(json_path, 'r') as f:
        qwen_data = json.load(f)
    
    # Convert to DataFrame for easier merging
    qwen_df = pd.DataFrame(qwen_data)
    qwen_df.rename(columns={'prediction': 'qwen_prediction'}, inplace=True)
    return qwen_df

def merge_predictions(csv_df, qwen_df):
    """Merge the CSV data with Qwen predictions"""
    merged_df = csv_df.merge(qwen_df, on='id', how='inner')
    return merged_df

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy and F1 score"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label='unhealthy')
    return accuracy, f1

def plot_metrics_comparison(llava_metrics, qwen_metrics):
    """Plot comparison of metrics between LLaVA-Med and Qwen"""
    models = ['LLaVA-Med', 'Qwen-2.5']
    accuracies = [llava_metrics[0], qwen_metrics[0]]
    f1_scores = [llava_metrics[1], qwen_metrics[1]]
    
    # Define colors
    colors = [ '#FFB347', '#87CEEB'] 
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Accuracy Comparison', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only for non-zero values)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1 Score plot
    bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('F1 Score Comparison', fontsize=18, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=16)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only for non-zero values)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_confusion_matrices(merged_df):
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # LLaVA-Med confusion matrix
    cm_llava = confusion_matrix(merged_df['label'], merged_df['prediction'])
    sns.heatmap(cm_llava, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Healthy', 'Unhealthy'], 
                yticklabels=['Healthy', 'Unhealthy'], ax=ax1,
                annot_kws={'size': 16})
    ax1.set_title('LLaVA-Med Confusion Matrix', fontweight='bold', fontsize=18)
    ax1.set_ylabel('True Label', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Qwen confusion matrix
    cm_qwen = confusion_matrix(merged_df['label'], merged_df['qwen_prediction'])
    sns.heatmap(cm_qwen, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Unhealthy'], 
                yticklabels=['Healthy', 'Unhealthy'], ax=ax2,
                annot_kws={'size': 16})
    ax2.set_title('Qwen-2.5 Confusion Matrix', fontweight='bold', fontsize=18)
    ax2.set_ylabel('True Label', fontsize=16)
    ax2.set_xlabel('Predicted Label', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()
    return fig

def main():
    # Paths to your files
    csv_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_classifications.csv"
    json_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\qwen\\qwen2_5_classification_results.json"
    
    print("Loading data...")
    
    # Load CSV data
    try:
        csv_df = load_csv_data(csv_path)
        print(f"Loaded CSV data with {len(csv_df)} samples")
    except FileNotFoundError:
        print("CSV file not found. Please check the path.")
        return None
    
    # Load Qwen predictions
    try:
        qwen_df = load_qwen_predictions(json_path)
        print(f"Loaded Qwen predictions for {len(qwen_df)} samples")
    except FileNotFoundError:
        print("Qwen predictions file not found. Please check the path.")
        return None
    
    # Merge the data
    merged_df = merge_predictions(csv_df, qwen_df)
    print(f"Merged data contains {len(merged_df)} samples")
    
    # Show sample of merged data
    print("\nSample of merged data:")
    print(merged_df.head())
    
    # Calculate metrics for both models
    llava_accuracy, llava_f1 = calculate_metrics(merged_df['label'], merged_df['prediction'])
    qwen_accuracy, qwen_f1 = calculate_metrics(merged_df['label'], merged_df['qwen_prediction'])
    
    print("\n=== Model Performance ===")
    print(f"LLaVA-Med - Accuracy: {llava_accuracy:.3f}, F1 Score: {llava_f1:.3f}")
    print(f"Qwen-2.5 - Accuracy: {qwen_accuracy:.3f}, F1 Score: {qwen_f1:.3f}")
    
    # Plot metrics comparison
    print("\nPlotting metrics comparison...")
    plot_metrics_comparison((llava_accuracy, llava_f1), (qwen_accuracy, qwen_f1))
    
    # Plot confusion matrices
    print("Plotting confusion matrices...")
    plot_confusion_matrices(merged_df)
    
    # Print detailed classification reports
    print("\n=== LLaVA-Med Classification Report ===")
    print(classification_report(merged_df['label'], merged_df['prediction']))
    
    print("\n=== Qwen-2.5 Classification Report ===")
    print(classification_report(merged_df['label'], merged_df['qwen_prediction']))
    
    # Save merged results
    output_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\merged_chest_classification_results.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged results saved to: {output_path}")

if __name__ == "__main__":
    main()