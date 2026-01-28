import pandas as pd
import numpy as np
import krippendorff
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load both holistic and turn-level data."""
    holistic_df = pd.read_csv('human_holistic.csv')
    turn_df = pd.read_csv('human_turn.csv')
    return holistic_df, turn_df

def reshape_for_krippendorff(df, group, metric, rating_type='holistic'):
    """
    Reshape data for Krippendorff's Alpha calculation.
    
    Parameters:
    - df: DataFrame with ratings
    - group: Group identifier ('Group A', 'Group B', 'Group C')
    - metric: Metric identifier (e.g., 'emotional_coherence')
    - rating_type: 'holistic' or 'turn'
    
    Returns:
    - Matrix suitable for Krippendorff's Alpha calculation
    """
    # Filter data for specific group and metric
    filtered_df = df[(df['group_id'] == group) & (df['metric_id'] == metric)].copy()
    
    if rating_type == 'turn':
        # For turn-level, we want turns 1-5
        filtered_df = filtered_df[filtered_df['turn'].between(1, 5)]
    
    if filtered_df.empty:
        return None
    
    # Create pivot table: rows = items (scenario, turn), columns = raters
    if rating_type == 'holistic':
        index_cols = ['scenario_id']
    else:  # turn
        index_cols = ['scenario_id', 'turn']
    
    pivot_table = filtered_df.pivot_table(
        index=index_cols,
        columns='rater_id',
        values='score',
        aggfunc='first'  # In case of duplicates
    )
    
    return pivot_table.values  # Return numpy array

def calculate_alpha_for_group_and_metric(df, group, metric, rating_type='holistic'):
    """
    Calculate Krippendorff's Alpha for a specific group and metric.
    """
    data_matrix = reshape_for_krippendorff(df, group, metric, rating_type)
    
    if data_matrix is None or data_matrix.size == 0:
        return None, 0, 0  # alpha, n_items, n_raters
    
    # Count non-NaN values to determine actual items and raters
    n_items, n_raters = data_matrix.shape
    valid_values = ~np.isnan(data_matrix)
    n_valid_ratings = np.sum(valid_values)
    
    if n_valid_ratings == 0:
        return None, 0, 0
    
    try:
        alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='interval')
        return alpha, n_items, n_raters
    except Exception as e:
        print(f"Error calculating alpha for {group}, {metric}: {e}")
        return None, n_items, n_raters

def run_comprehensive_analysis():
    """Run comprehensive Krippendorff's Alpha analysis."""
    print("Loading data...")
    holistic_df, turn_df = load_data()
    
    groups = ['Group A', 'Group B', 'Group C']
    holistic_metrics = sorted(holistic_df['metric_id'].unique())
    turn_metrics = sorted(turn_df['metric_id'].unique())
    
    print(f"\nGroups: {groups}")
    print(f"Holistic metrics: {holistic_metrics}")
    print(f"Turn metrics: {turn_metrics}")
    
    # Analysis results
    results = {
        'holistic': [],
        'turn': []
    }
    
    print("\n" + "="*60)
    print("KRIPPENDORFF'S ALPHA ANALYSIS RESULTS")
    print("="*60)
    
    # Holistic Analysis
    print("\n1. HOLISTIC LEVEL ANALYSIS")
    print("-" * 40)
    
    for group in groups:
        print(f"\n{group}:")
        group_holistic_results = []
        
        for metric in holistic_metrics:
            alpha, n_items, n_raters = calculate_alpha_for_group_and_metric(
                holistic_df, group, metric, 'holistic'
            )
            
            result = {
                'group': group,
                'metric': metric,
                'alpha': alpha,
                'n_items': n_items,
                'n_raters': n_raters,
                'type': 'holistic'
            }
            
            results['holistic'].append(result)
            group_holistic_results.append(result)
            
            if alpha is not None:
                print(f"  {metric:<25} α = {alpha:.3f} (items: {n_items}, raters: {n_raters})")
            else:
                print(f"  {metric:<25} α = N/A (no data)")
    
    # Turn-level Analysis
    print("\n\n2. TURN LEVEL ANALYSIS")
    print("-" * 40)
    
    for group in groups:
        print(f"\n{group}:")
        group_turn_results = []
        
        for metric in turn_metrics:
            alpha, n_items, n_raters = calculate_alpha_for_group_and_metric(
                turn_df, group, metric, 'turn'
            )
            
            result = {
                'group': group,
                'metric': metric,
                'alpha': alpha,
                'n_items': n_items,
                'n_raters': n_raters,
                'type': 'turn'
            }
            
            results['turn'].append(result)
            group_turn_results.append(result)
            
            if alpha is not None:
                print(f"  {metric:<25} α = {alpha:.3f} (items: {n_items}, raters: {n_raters})")
            else:
                print(f"  {metric:<25} α = N/A (no data)")
    
    # Overall analysis
    print("\n\n3. SUMMARY STATISTICS")
    print("-" * 40)
    
    for analysis_type, data_list in results.items():
        print(f"\n{analysis_type.upper()} Ratings:")
        
        # Filter out None values for statistics
        valid_alphas = [r['alpha'] for r in data_list if r['alpha'] is not None]
        
        if valid_alphas:
            avg_alpha = np.mean(valid_alphas)
            std_alpha = np.std(valid_alphas)
            min_alpha = np.min(valid_alphas)
            max_alpha = np.max(valid_alphas)
            
            print(f"  Average α: {avg_alpha:.3f} (±{std_alpha:.3f})")
            print(f"  Range: {min_alpha:.3f} - {max_alpha:.3f}")
            print(f"  Total metrics analyzed: {len(valid_alphas)}")
        else:
            print("  No valid alphas to analyze")
    
    # Interpretation guide
    print("\n\n4. INTERPRETATION GUIDE")
    print("-" * 40)
    print("α ≥ 0.80: Reliable (for group comparisons)")
    print("α ≥ 0.67: Minimally acceptable (for individual assessments)")
    print("α < 0.67: Unacceptable reliability")
    
    return results

def detailed_report(results):
    """Generate a detailed report of the findings."""
    print("\n\n" + "="*80)
    print("DETAILED RELIABILITY ANALYSIS REPORT")
    print("="*80)
    
    for analysis_type in ['holistic', 'turn']:
        print(f"\n{analysis_type.upper().replace('_', ' ')} ANALYSIS DETAILED RESULTS")
        print("-" * 60)
        
        type_results = [r for r in results[analysis_type] if r['alpha'] is not None]
        
        if not type_results:
            print("No valid results for this analysis type.")
            continue
            
        # Sort by alpha value
        sorted_results = sorted(type_results, key=lambda x: x['alpha'] if x['alpha'] is not None else float('-inf'), reverse=True)
        
        print(f"{'Group':<10} {'Metric':<25} {'Alpha':<8} {'Items':<6} {'Raters':<7} {'Reliability'}")
        print("-" * 80)
        
        for result in sorted_results:
            alpha = result['alpha']
            if alpha is not None:
                if alpha >= 0.80:
                    reliability = "High ✓"
                elif alpha >= 0.67:
                    reliability = "Acceptable ◐"
                else:
                    reliability = "Low ✗"
                
                print(f"{result['group']:<10} {result['metric']:<25} {alpha:<8.3f} "
                      f"{result['n_items']:<6} {result['n_raters']:<7} {reliability}")
            else:
                print(f"{result['group']:<10} {result['metric']:<25} {'N/A':<8} "
                      f"{result['n_items']:<6} {result['n_raters']:<7} {'No Data'}")
    
    # Reliability by group
    print(f"\n\nRELIABILITY BY GROUP")
    print("-" * 60)
    
    for group in ['Group A', 'Group B', 'Group C']:
        print(f"\n{group}:")
        
        # Holistic alphas for this group
        holistic_alphas = [r['alpha'] for r in results['holistic'] 
                          if r['group'] == group and r['alpha'] is not None]
        if holistic_alphas:
            avg_holistic = np.mean(holistic_alphas)
            print(f"  Holistic α: {avg_holistic:.3f} (n={len(holistic_alphas)} metrics)")
        
        # Turn alphas for this group
        turn_alphas = [r['alpha'] for r in results['turn'] 
                      if r['group'] == group and r['alpha'] is not None]
        if turn_alphas:
            avg_turn = np.mean(turn_alphas)
            print(f"  Turn-level α: {avg_turn:.3f} (n={len(turn_alphas)} metrics)")

if __name__ == "__main__":
    results = run_comprehensive_analysis()
    detailed_report(results)
    
    print(f"\n\nAnalysis completed! The results show the inter-rater reliability using Krippendorff's Alpha.")
    print("Higher values (closer to 1.0) indicate greater agreement among raters.")