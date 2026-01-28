import pandas as pd
import numpy as np
import krippendorff
from scipy.stats import pearsonr
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

def reshape_for_icc(df, group, metric, rating_type='holistic'):
    """
    Reshape data for ICC calculation.
    """
    # Filter data for specific group and metric
    filtered_df = df[(df['group_id'] == group) & (df['metric_id'] == metric)].copy()
    
    if rating_type == 'turn':
        # For turn-level, we want turns 1-5
        filtered_df = filtered_df[filtered_df['turn'].between(1, 5)]
    
    if filtered_df.empty:
        return None
    
    # Create pivot table: rows = raters, columns = items (scenario, turn)
    if rating_type == 'holistic':
        index_cols = 'rater_id'
        column_cols = 'scenario_id'
    else:  # turn
        index_cols = 'rater_id'
        column_cols = ['scenario_id', 'turn']
    
    pivot_table = filtered_df.pivot_table(
        index=index_cols,
        columns=column_cols,
        values='score',
        aggfunc='first'  # In case of duplicates
    )
    
    return pivot_table.values  # Return numpy array

def icc_two_way_single(ratings):
    """
    Calculate ICC(2,1) - consistency of single raters using a two-way model.
    """
    # Remove rows with all NaN values
    ratings = ratings[~np.all(np.isnan(ratings), axis=1)]
    if ratings.shape[0] < 2:
        return np.nan
    
    # Remove columns with all NaN values
    ratings = ratings[:, ~np.all(np.isnan(ratings), axis=0)]
    if ratings.shape[1] < 2:
        return np.nan
    
    # Replace remaining NaN values with column means (for calculation purposes)
    col_means = np.nanmean(ratings, axis=0)
    for j in range(ratings.shape[1]):
        nan_mask = np.isnan(ratings[:, j])
        ratings[nan_mask, j] = col_means[j]
    
    n_raters, n_items = ratings.shape
    
    # Calculate means
    item_means = np.mean(ratings, axis=0)
    rater_means = np.mean(ratings, axis=1)
    grand_mean = np.mean(ratings)
    
    # Calculate sums of squares
    ss_item = n_raters * np.sum((item_means - grand_mean)**2)
    ss_rater = n_items * np.sum((rater_means - grand_mean)**2)
    ss_error = 0
    for i in range(n_raters):
        for j in range(n_items):
            ss_error += (ratings[i, j] - item_means[j] - rater_means[i] + grand_mean)**2
    
    # Mean squares
    ms_item = ss_item / (n_items - 1) if n_items > 1 else 0
    ms_rater = ss_rater / (n_raters - 1) if n_raters > 1 else 0
    ms_error = ss_error / ((n_raters - 1) * (n_items - 1)) if n_raters > 1 and n_items > 1 else 0
    
    # ICC(2,1) formula
    if ms_error == 0 and ms_rater == 0:
        return 1.0
    elif (ms_item - ms_error) <= 0:
        return 0.0
    else:
        icc_value = (ms_item - ms_error) / (ms_item + (n_raters - 1) * ms_error + n_raters * (ms_rater - ms_error) / n_items)
        return icc_value

def calculate_icc_for_group_and_metric(df, group, metric, rating_type='holistic'):
    """
    Calculate ICC for a specific group and metric.
    """
    data_matrix = reshape_for_icc(df, group, metric, rating_type)
    
    if data_matrix is None or data_matrix.size == 0 or data_matrix.shape[0] < 2 or data_matrix.shape[1] < 2:
        return None, 0, 0  # icc, n_items, n_raters
    
    # Transpose if needed to ensure raters are rows and items are columns
    if data_matrix.shape[0] > data_matrix.shape[1]:
        data_matrix = data_matrix.T
    
    try:
        icc = icc_two_way_single(data_matrix)
        n_raters, n_items = data_matrix.shape
        return icc, n_items, n_raters
    except Exception as e:
        print(f"Error calculating ICC for {group}, {metric}: {e}")
        return None, 0, 0

def run_comparison_analysis():
    """Run comprehensive comparison analysis of Krippendorff's Alpha and ICC."""
    print("Loading data...")
    holistic_df, turn_df = load_data()
    
    groups = ['Group A', 'Group B', 'Group C']
    holistic_metrics = sorted(holistic_df['metric_id'].unique())
    turn_metrics = sorted(turn_df['metric_id'].unique())
    
    print(f"\nGroups: {groups}")
    print(f"Holistic metrics: {holistic_metrics}")
    print(f"Turn metrics: {turn_metrics}")
    
    # Combined results
    results = {
        'holistic': [],
        'turn': []
    }
    
    print("\n" + "="*100)
    print("COMPARISON ANALYSIS: KRIPPENDORFF'S ALPHA VS ICC")
    print("="*100)
    
    # Holistic Analysis
    print("\n1. HOLISTIC LEVEL ANALYSIS")
    print("-" * 60)
    print(f"{'Group':<10} {'Metric':<25} {'Alpha':<10} {'ICC':<10} {'Items':<6} {'Raters':<7}")
    print("-" * 85)
    
    for group in groups:
        for metric in holistic_metrics:
            alpha, n_items_a, n_raters_a = calculate_alpha_for_group_and_metric(
                holistic_df, group, metric, 'holistic'
            )
            
            icc, n_items_i, n_raters_i = calculate_icc_for_group_and_metric(
                holistic_df, group, metric, 'holistic'
            )
            
            result = {
                'group': group,
                'metric': metric,
                'alpha': alpha,
                'icc': icc,
                'n_items': n_items_a,
                'n_raters': n_raters_a,
                'type': 'holistic'
            }
            
            results['holistic'].append(result)
            
            alpha_str = f"{alpha:.3f}" if alpha is not None else "N/A"
            icc_str = f"{icc:.3f}" if icc is not None and not np.isnan(icc) else "N/A"
            
            print(f"{group:<10} {metric:<25} {alpha_str:<10} {icc_str:<10} {n_items_a:<6} {n_raters_a:<7}")
    
    # Turn-level Analysis
    print("\n\n2. TURN LEVEL ANALYSIS")
    print("-" * 60)
    print(f"{'Group':<10} {'Metric':<25} {'Alpha':<10} {'ICC':<10} {'Items':<6} {'Raters':<7}")
    print("-" * 85)
    
    for group in groups:
        for metric in turn_metrics:
            alpha, n_items_a, n_raters_a = calculate_alpha_for_group_and_metric(
                turn_df, group, metric, 'turn'
            )
            
            icc, n_items_i, n_raters_i = calculate_icc_for_group_and_metric(
                turn_df, group, metric, 'turn'
            )
            
            result = {
                'group': group,
                'metric': metric,
                'alpha': alpha,
                'icc': icc,
                'n_items': n_items_a,
                'n_raters': n_raters_a,
                'type': 'turn'
            }
            
            results['turn'].append(result)
            
            alpha_str = f"{alpha:.3f}" if alpha is not None else "N/A"
            icc_str = f"{icc:.3f}" if icc is not None and not np.isnan(icc) else "N/A"
            
            print(f"{group:<10} {metric:<25} {alpha_str:<10} {icc_str:<10} {n_items_a:<6} {n_raters_a:<7}")
    
    # Summary statistics
    print("\n\n3. SUMMARY STATISTICS")
    print("-" * 60)
    
    for analysis_type in ['holistic', 'turn']:
        print(f"\n{analysis_type.upper()} Ratings:")
        
        # Filter out None/NaN values for statistics
        valid_alphas = [r['alpha'] for r in results[analysis_type] 
                       if r['alpha'] is not None]
        valid_iccs = [r['icc'] for r in results[analysis_type] 
                     if r['icc'] is not None and not np.isnan(r['icc'])]
        
        if valid_alphas:
            avg_alpha = np.mean(valid_alphas)
            std_alpha = np.std(valid_alphas)
            min_alpha = np.min(valid_alphas)
            max_alpha = np.max(valid_alphas)
            
            print(f"  Average Alpha: {avg_alpha:.3f} (±{std_alpha:.3f})")
            print(f"  Alpha Range: {min_alpha:.3f} - {max_alpha:.3f}")
        
        if valid_iccs:
            avg_icc = np.mean(valid_iccs)
            std_icc = np.std(valid_iccs)
            min_icc = np.min(valid_iccs)
            max_icc = np.max(valid_iccs)
            
            print(f"  Average ICC: {avg_icc:.3f} (±{std_icc:.3f})")
            print(f"  ICC Range: {min_icc:.3f} - {max_icc:.3f}")
    
    # Correlation between Alpha and ICC
    print("\n\n4. CORRELATION BETWEEN ALPHA AND ICC")
    print("-" * 60)
    
    for analysis_type in ['holistic', 'turn']:
        type_results = results[analysis_type]
        
        # Get pairs where both alpha and icc are valid
        valid_pairs = [(r['alpha'], r['icc']) for r in type_results 
                      if r['alpha'] is not None and r['icc'] is not None and not np.isnan(r['icc'])]
        
        if len(valid_pairs) > 1:
            alphas, iccs = zip(*valid_pairs)
            correlation, p_value = pearsonr(alphas, iccs)
            print(f"{analysis_type.capitalize()} ratings correlation: r = {correlation:.3f}, p = {p_value:.3f}")
        else:
            print(f"Not enough valid pairs for {analysis_type} ratings to calculate correlation")
    
    # Reliability thresholds comparison
    print("\n\n5. RELIABILITY THRESHOLD COMPARISONS")
    print("-" * 60)
    
    for analysis_type in ['holistic', 'turn']:
        print(f"\n{analysis_type.upper()} Ratings:")
        
        # Count how many meet different thresholds
        type_results = results[analysis_type]
        
        # Alpha thresholds
        alpha_reliable = sum(1 for r in type_results 
                            if r['alpha'] is not None and r['alpha'] >= 0.8)
        alpha_acceptable = sum(1 for r in type_results 
                              if r['alpha'] is not None and r['alpha'] >= 0.67)
        
        # ICC thresholds
        icc_excellent = sum(1 for r in type_results 
                           if r['icc'] is not None and not np.isnan(r['icc']) and r['icc'] >= 0.75)
        icc_good = sum(1 for r in type_results 
                      if r['icc'] is not None and not np.isnan(r['icc']) and r['icc'] >= 0.6)
        
        print(f"  Alpha ≥ 0.80 (Reliable): {alpha_reliable}/{len(type_results)} metrics")
        print(f"  Alpha ≥ 0.67 (Acceptable): {alpha_acceptable}/{len(type_results)} metrics")
        print(f"  ICC ≥ 0.75 (Excellent): {icc_excellent}/{len(type_results)} metrics")
        print(f"  ICC ≥ 0.60 (Good): {icc_good}/{len(type_results)} metrics")
    
    return results

def detailed_comparison_report(results):
    """Generate a detailed comparison report."""
    print("\n\n" + "="*100)
    print("DETAILED COMPARISON REPORT")
    print("="*100)
    
    for analysis_type in ['holistic', 'turn']:
        print(f"\n{analysis_type.upper().replace('_', ' ')} ANALYSIS DETAILED COMPARISON")
        print("-" * 80)
        
        type_results = [r for r in results[analysis_type] 
                       if r['alpha'] is not None and r['icc'] is not None and not np.isnan(r['icc'])]
        
        if not type_results:
            print("No valid results for this analysis type.")
            continue
        
        print(f"{'Group':<10} {'Metric':<25} {'Alpha':<8} {'ICC':<8} {'Alpha_Reliability':<15} {'ICC_Reliability':<15}")
        print("-" * 95)
        
        for result in type_results:
            alpha = result['alpha']
            icc = result['icc']
            
            # Determine reliability categories
            if alpha >= 0.8:
                alpha_rel = "Reliable"
            elif alpha >= 0.67:
                alpha_rel = "Acceptable"
            else:
                alpha_rel = "Low"
            
            if icc >= 0.75:
                icc_rel = "Excellent"
            elif icc >= 0.6:
                icc_rel = "Good"
            elif icc >= 0.4:
                icc_rel = "Fair"
            else:
                icc_rel = "Poor"
            
            print(f"{result['group']:<10} {result['metric']:<25} {alpha:<8.3f} "
                  f"{icc:<8.3f} {alpha_rel:<15} {icc_rel:<15}")

if __name__ == "__main__":
    results = run_comparison_analysis()
    detailed_comparison_report(results)
    
    print(f"\n\nComparison analysis completed!")
    print("Both Krippendorff's Alpha and ICC provide insights into inter-rater reliability.")
    print("Alpha is more robust to missing data and different measurement levels.")
    print("ICC is more traditional and may be more familiar to some audiences.")