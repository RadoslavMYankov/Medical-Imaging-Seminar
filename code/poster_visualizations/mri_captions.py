import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_mri_captions_csv(csv_path):
    """Load the MRI captions CSV file"""
    df = pd.read_csv(csv_path)
    return df

def load_qwen_llm_scores(json_path):
    """Load Qwen LLM scores from JSON file"""
    with open(json_path, 'r') as f:
        qwen_data = json.load(f)
    
    # Convert to DataFrame for easier merging
    qwen_df = pd.DataFrame(qwen_data)
    
    # Keep only rows with actual scores (filter out entries without predictions/scores)
    qwen_df = qwen_df.dropna(subset=['llm_score'])
    
    return qwen_df

def merge_captions_with_qwen_scores(mri_df, qwen_df):
    """Merge the MRI captions with Qwen LLM scores"""
    # Merge on image name
    merged_df = mri_df.merge(
        qwen_df[['image', 'llm_score']], 
        on='image', 
        how='inner',
        suffixes=('', '_qwen')
    )
    
    # Rename columns for clarity
    merged_df = merged_df.rename(columns={
        'gpt_rating': 'llava_llm_score',
        'llm_score': 'qwen_llm_score'
    })
    
    return merged_df

def plot_average_llm_scores(merged_df):
    """Plot average LLM scores comparison"""
    llava_mean = merged_df['llava_llm_score'].mean()
    qwen_mean = merged_df['qwen_llm_score'].mean()
    
    models = ['LLaVA-Med', 'Qwen-2.5']
    scores = [llava_mean, qwen_mean]
    
    # Define colors - light blue and light orange
    colors = ['#87CEEB', '#FFB347']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    plt.title('Average LLM Rating Comparison for MRI Captions', fontsize=14, fontweight='bold')
    plt.ylabel('Average LLM Rating', fontsize=12)
    plt.ylim(0, max(scores) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only for non-zero values)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()

def plot_combined_histogram(merged_df):
    """Plot combined histogram showing both models side by side"""
    # Prepare data for plotting
    plot_data = []
    
    for _, row in merged_df.iterrows():
        plot_data.append({'Model': 'LLaVA-Med', 'Score': row['llava_llm_score']})
        plot_data.append({'Model': 'Qwen-2.5', 'Score': row['qwen_llm_score']})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define colors - orange for LLaVA-Med, light blue for Qwen-2.5
    colors = ['#FFB347', '#87CEEB']
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped histogram with thinner bars
    ax = sns.histplot(data=plot_df, x='Score', hue='Model', 
                      bins=np.arange(0.5, 5.5, 1), alpha=0.8, multiple='dodge',
                      palette=colors, edgecolor='black', linewidth=1,
                      shrink=0.6) 
    
    plt.title('LLM Score Distribution Comparison for MRI Captions', fontsize=14, fontweight='bold')
    plt.xlabel('LLM Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Set x-axis ticks in the middle of the grouped bars
    plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4'])
    plt.xlim(0.5, 4.5)
    
    plt.grid(axis='y', alpha=0.3)
    
    # Ensure legend shows with correct labels
    plt.legend(title='Model', labels=['Qwen-2.5', 'LLaVA-Med'], fontsize=10)
    
    # Add count labels on bars
    for container in ax.containers:
        labels = []
        for v in container:
            height = v.get_height()
            if height > 0:
                labels.append(f'{int(height)}')
            else:
                labels.append('')
        ax.bar_label(container, labels=labels, fontsize=9)
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()

def create_detailed_comparison_table(merged_df):
    """Create a detailed comparison table"""
    # Calculate statistics for both models
    llava_stats = {
        'Mean': merged_df['llava_llm_score'].mean(),
        'Median': merged_df['llava_llm_score'].median(),
        'Std': merged_df['llava_llm_score'].std(),
        'Min': merged_df['llava_llm_score'].min(),
        'Max': merged_df['llava_llm_score'].max()
    }
    
    qwen_stats = {
        'Mean': merged_df['qwen_llm_score'].mean(),
        'Median': merged_df['qwen_llm_score'].median(),
        'Std': merged_df['qwen_llm_score'].std(),
        'Min': merged_df['qwen_llm_score'].min(),
        'Max': merged_df['qwen_llm_score'].max()
    }
    
    # Create comparison table
    comparison_data = []
    for metric in llava_stats.keys():
        comparison_data.append({
            'Metric': metric,
            'LLaVA-Med': f"{llava_stats[metric]:.3f}",
            'Qwen-2.5': f"{qwen_stats[metric]:.3f}",
            'Difference': f"{qwen_stats[metric] - llava_stats[metric]:.3f}"
        })
    
    # Add score distribution
    for score in range(1, 5):
        llava_count = (merged_df['llava_llm_score'] == score).sum()
        qwen_count = (merged_df['qwen_llm_score'] == score).sum()
        
        comparison_data.append({
            'Metric': f'Score {score} Count',
            'LLaVA-Med': str(llava_count),
            'Qwen-2.5': str(qwen_count),
            'Difference': str(qwen_count - llava_count)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def save_merged_results(merged_df, output_path):
    """Save the merged results to CSV"""
    merged_df.to_csv(output_path, index=False)
    print(f"Merged MRI captions with LLM scores saved to: {output_path}")

def main():
    """Main function to run the MRI caption analysis"""
    # Paths to your files
    mri_csv_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_captions.csv"
    qwen_json_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\qwen\\llm_eval_per_image_qwen2_5_description_results_zero_shot.json_v2.json"

    print("Loading MRI captions and Qwen LLM scores...")
    
    try:
        # Load data
        mri_df = load_mri_captions_csv(mri_csv_path)
        qwen_df = load_qwen_llm_scores(qwen_json_path)
        
        print(f"Loaded MRI captions with {len(mri_df)} samples")
        print(f"Loaded Qwen LLM scores for {len(qwen_df)} samples")
        
        # Merge data
        merged_df = merge_captions_with_qwen_scores(mri_df, qwen_df)
        print(f"Merged data contains {len(merged_df)} samples")
        
        # Display sample of merged data
        print("\nSample of merged data:")
        print(merged_df[['image', 'llava_llm_score', 'qwen_llm_score']].head())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Calculate and display statistics
    print(f"\n=== LLM Score Statistics ===")
    print(f"LLaVA-Med - Mean: {merged_df['llava_llm_score'].mean():.3f}, Std: {merged_df['llava_llm_score'].std():.3f}")
    print(f"Qwen-2.5 - Mean: {merged_df['qwen_llm_score'].mean():.3f}, Std: {merged_df['qwen_llm_score'].std():.3f}")
    
    # Create detailed comparison table
    comparison_table = create_detailed_comparison_table(merged_df)
    print("\n=== Detailed Comparison ===")
    print(comparison_table.to_string(index=False))
    
    # Create visualizations
    print("\nPlotting average LLM scores...")
    plot_average_llm_scores(merged_df)
    print("\nPlotting combined histogram...")
    plot_combined_histogram(merged_df)
    
    # Save results
    print("\nSaving results...")
    merged_output_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_captions_with_qwen_scores.csv"
    save_merged_results(merged_df, merged_output_path)

    comparison_output_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_llm_score_comparison.csv"
    comparison_table.to_csv(comparison_output_path, index=False)
    print(f"Comparison table saved to: {comparison_output_path}")

if __name__ == "__main__":
    main()