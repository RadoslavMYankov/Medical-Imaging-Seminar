import pandas as pd
import json

def load_llava_bleu_scores(csv_path):
    """Load LLaVA-Med BLEU scores from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def load_qwen_bleu_scores(json_path):
    """Load Qwen-2.5 BLEU scores from JSON file"""
    with open(json_path, 'r') as f:
        qwen_data = json.load(f)
    
    # Convert to DataFrame for easier processing
    qwen_df = pd.DataFrame(qwen_data)
    
    # Keep only rows with BLEU scores (filter out entries without predictions)
    qwen_df = qwen_df.dropna(subset=['bleu1'])
    
    return qwen_df

def merge_bleu_scores(llava_df, qwen_df):
    """Merge LLaVA-Med and Qwen-2.5 BLEU scores"""
    # Merge on image name
    merged_df = llava_df.merge(
        qwen_df[['image', 'bleu1', 'bleu2', 'bleu3', 'bleu4']], 
        on='image', 
        how='inner',
        suffixes=('_llava', '_qwen')
    )
    
    return merged_df

def calculate_bleu_statistics(merged_df):
    """Calculate mean and standard deviation for BLEU scores"""
    bleu_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    
    statistics = {}
    
    for metric in bleu_metrics:
        llava_col = f'{metric}_llava'
        qwen_col = f'{metric}_qwen'
        
        # Calculate statistics for LLaVA-Med
        llava_mean = merged_df[llava_col].mean()
        llava_std = merged_df[llava_col].std()
        
        # Calculate statistics for Qwen-2.5
        qwen_mean = merged_df[qwen_col].mean()
        qwen_std = merged_df[qwen_col].std()
        
        statistics[metric] = {
            'llava_mean': llava_mean,
            'llava_std': llava_std,
            'qwen_mean': qwen_mean,
            'qwen_std': qwen_std,
            'llava_better': llava_mean > qwen_mean
        }
    
    return statistics

def format_score_with_bold(mean, std, is_better):
    """Format score with ± and bold if it's better"""
    formatted = f"{mean:.4f} ± {std:.4f}"
    if is_better:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted

def generate_latex_table(statistics):
    """Generate LaTeX table with BLEU score comparison"""
    
    # Table header
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{BLEU Score Comparison between LLaVA-Med and Qwen-2.5 for MRI Caption Generation}
\\label{tab:bleu_comparison}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Metric} & \\textbf{LLaVA-Med} & \\textbf{Qwen-2.5} \\\\
\\midrule
"""
    
    # Add rows for each BLEU metric
    bleu_names = {
        'bleu1': 'BLEU-1',
        'bleu2': 'BLEU-2', 
        'bleu3': 'BLEU-3',
        'bleu4': 'BLEU-4'
    }
    
    for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        stat = statistics[metric]
        
        # Format LLaVA-Med score
        llava_formatted = format_score_with_bold(
            stat['llava_mean'], 
            stat['llava_std'], 
            stat['llava_better']
        )
        
        # Format Qwen-2.5 score  
        qwen_formatted = format_score_with_bold(
            stat['qwen_mean'], 
            stat['qwen_std'], 
            not stat['llava_better']
        )
        
        latex_table += f"{bleu_names[metric]} & {llava_formatted} & {qwen_formatted} \\\\\n"
    
    # Table footer
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def create_detailed_comparison_table(statistics, merged_df):
    """Create a detailed comparison table for analysis"""
    comparison_data = []
    
    for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        stat = statistics[metric]
        
        comparison_data.append({
            'Metric': metric.upper(),
            'LLaVA-Med Mean': f"{stat['llava_mean']:.6f}",
            'LLaVA-Med Std': f"{stat['llava_std']:.6f}",
            'Qwen-2.5 Mean': f"{stat['qwen_mean']:.6f}",
            'Qwen-2.5 Std': f"{stat['qwen_std']:.6f}",
            'Better Model': 'LLaVA-Med' if stat['llava_better'] else 'Qwen-2.5',
            'Difference': f"{stat['qwen_mean'] - stat['llava_mean']:.6f}",
            'Sample Count': len(merged_df)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def save_results(latex_table, comparison_df, merged_df, base_path):
    """Save all results to files"""
    
    # Save LaTeX table
    latex_path = f"{base_path}_latex_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")
    
    # Save detailed comparison
    comparison_path = f"{base_path}_detailed_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Detailed comparison saved to: {comparison_path}")
    
    # Save merged data
    merged_path = f"{base_path}_merged_bleu_scores.csv"
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged BLEU scores saved to: {merged_path}")

def main():
    """Main function to run the BLEU score analysis"""
    # Paths to your files
    llava_csv_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_bleu_scores.csv"
    qwen_json_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\qwen\\bleu_per_image.json"
    
    print("Loading BLEU scores...")
    
    try:
        # Load data
        llava_df = load_llava_bleu_scores(llava_csv_path)
        qwen_df = load_qwen_bleu_scores(qwen_json_path)
        
        print(f"Loaded LLaVA-Med BLEU scores for {len(llava_df)} images")
        print(f"Loaded Qwen-2.5 BLEU scores for {len(qwen_df)} images")
        
        # Merge data
        merged_df = merge_bleu_scores(llava_df, qwen_df)
        print(f"Merged data contains {len(merged_df)} images")
        
        # Display sample of merged data
        print("\nSample of merged data:")
        print(merged_df[['image', 'bleu1_llava', 'bleu1_qwen', 'bleu2_llava', 'bleu2_qwen']].head())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Calculate statistics
    print("\nCalculating BLEU statistics...")
    statistics = calculate_bleu_statistics(merged_df)
    
    # Display results
    print("\n=== BLEU Score Statistics ===")
    for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        stat = statistics[metric]
        better_model = "LLaVA-Med" if stat['llava_better'] else "Qwen-2.5"
        print(f"\n{metric.upper()}:")
        print(f"  LLaVA-Med: {stat['llava_mean']:.6f} ± {stat['llava_std']:.6f}")
        print(f"  Qwen-2.5:  {stat['qwen_mean']:.6f} ± {stat['qwen_std']:.6f}")
        print(f"  Better: {better_model}")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(statistics)
    print("\n=== LaTeX Table ===")
    print(latex_table)
    
    # Create detailed comparison table
    comparison_df = create_detailed_comparison_table(statistics, merged_df)
    print("\n=== Detailed Comparison ===")
    print(comparison_df.to_string(index=False))
    
    # Save results
    print("\nSaving results...")
    base_path = "C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\bleu_comparison"
    save_results(latex_table, comparison_df, merged_df, base_path)

if __name__ == "__main__":
    main()
