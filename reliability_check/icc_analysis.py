import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load both holistic and turn-level data."""
    holistic_df = pd.read_csv('human_holistic.csv')
    turn_df = pd.read_csv('human_turn.csv')
    return holistic_df, turn_df

def reshape_for_icc(df, group, metric, rating_type='holistic'):
    """
    Reshape data for ICC calculation.
    
    Parameters:
    - df: DataFrame with ratings
    - group: Group identifier ('Group A', 'Group B', 'Group C')
    - metric: Metric identifier (e.g., 'emotional_coherence')
    - rating_type: 'holistic' or 'turn'
    
    Returns:
    - Matrix suitable for ICC calculation (raters x items)
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
    
    Parameters:
    - ratings: 2D array with shape (n_raters, n_items)
    
    Returns:
    - ICC value
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

def run_icc_analysis():
    """Run comprehensive ICC analysis."""
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
    print("INTRACLASS CORRELATION COEFFICIENT (ICC) ANALYSIS RESULTS")
    print("="*60)
    
    # Holistic Analysis
    print("\n1. HOLISTIC LEVEL ANALYSIS")
    print("-" * 40)
    
    for group in groups:
        print(f"\n{group}:")
        
        for metric in holistic_metrics:
            icc, n_items, n_raters = calculate_icc_for_group_and_metric(
                holistic_df, group, metric, 'holistic'
            )
            
            result = {
                'group': group,
                'metric': metric,
                'icc': icc,
                'n_items': n_items,
                'n_raters': n_raters,
                'type': 'holistic'
            }
            
            results['holistic'].append(result)
            
            if icc is not None and not np.isnan(icc):
                print(f"  {metric:<25} ICC = {icc:.3f} (items: {n_items}, raters: {n_raters})")
            else:
                print(f"  {metric:<25} ICC = N/A (insufficient data)")
    
    # Turn-level Analysis
    print("\n\n2. TURN LEVEL ANALYSIS")
    print("-" * 40)
    
    for group in groups:
        print(f"\n{group}:")
        
        for metric in turn_metrics:
            icc, n_items, n_raters = calculate_icc_for_group_and_metric(
                turn_df, group, metric, 'turn'
            )
            
            result = {
                'group': group,
                'metric': metric,
                'icc': icc,
                'n_items': n_items,
                'n_raters': n_raters,
                'type': 'turn'
            }
            
            results['turn'].append(result)
            
            if icc is not None and not np.isnan(icc):
                print(f"  {metric:<25} ICC = {icc:.3f} (items: {n_items}, raters: {n_raters})")
            else:
                print(f"  {metric:<25} ICC = N/A (insufficient data)")
    
    # Overall analysis
    print("\n\n3. SUMMARY STATISTICS")
    print("-" * 40)
    
    for analysis_type, data_list in results.items():
        print(f"\n{analysis_type.upper()} Ratings:")
        
        # Filter out None and NaN values for statistics
        valid_iccs = [r['icc'] for r in data_list if r['icc'] is not None and not np.isnan(r['icc'])]
        
        if valid_iccs:
            avg_icc = np.mean(valid_iccs)
            std_icc = np.std(valid_iccs)
            min_icc = np.min(valid_iccs)
            max_icc = np.max(valid_iccs)
            
            print(f"  Average ICC: {avg_icc:.3f} (±{std_icc:.3f})")
            print(f"  Range: {min_icc:.3f} - {max_icc:.3f}")
            print(f"  Total metrics analyzed: {len(valid_iccs)}")
        else:
            print("  No valid ICCs to analyze")
    
    # Interpretation guide
    print("\n\n4. INTERPRETATION GUIDE")
    print("-" * 40)
    print("ICC ≥ 0.75: Excellent reliability")
    print("ICC 0.60-0.74: Good reliability") 
    print("ICC 0.40-0.59: Fair reliability")
    print("ICC < 0.40: Poor reliability")
    
    return results

def detailed_icc_report(results):
    """Generate a detailed ICC report of the findings."""
    print("\n\n" + "="*80)
    print("DETAILED ICC ANALYSIS REPORT")
    print("="*80)
    
    for analysis_type in ['holistic', 'turn']:
        print(f"\n{analysis_type.upper().replace('_', ' ')} ANALYSIS DETAILED RESULTS")
        print("-" * 60)
        
        type_results = [r for r in results[analysis_type] if r['icc'] is not None and not np.isnan(r['icc'])]
        
        if not type_results:
            print("No valid results for this analysis type.")
            continue
            
        # Sort by ICC value
        sorted_results = sorted(type_results, key=lambda x: x['icc'] if x['icc'] is not None and not np.isnan(x['icc']) else float('-inf'), reverse=True)
        
        print(f"{'Group':<10} {'Metric':<25} {'ICC':<8} {'Items':<6} {'Raters':<7} {'Reliability'}")
        print("-" * 80)
        
        for result in sorted_results:
            icc = result['icc']
            if icc is not None and not np.isnan(icc):
                if icc >= 0.75:
                    reliability = "Excellent ✓"
                elif icc >= 0.60:
                    reliability = "Good ◐"
                elif icc >= 0.40:
                    reliability = "Fair △"
                else:
                    reliability = "Poor ✗"
                
                print(f"{result['group']:<10} {result['metric']:<25} {icc:<8.3f} "
                      f"{result['n_items']:<6} {result['n_raters']:<7} {reliability}")
            else:
                print(f"{result['group']:<10} {result['metric']:<25} {'N/A':<8} "
                      f"{result['n_items']:<6} {result['n_raters']:<7} {'No Data'}")
    
    # Reliability by group
    print(f"\n\nRELIABILITY BY GROUP")
    print("-" * 60)
    
    for group in ['Group A', 'Group B', 'Group C']:
        print(f"\n{group}:")
        
        # Holistic ICCs for this group
        holistic_iccs = [r['icc'] for r in results['holistic'] 
                         if r['group'] == group and r['icc'] is not None and not np.isnan(r['icc'])]
        if holistic_iccs:
            avg_holistic = np.mean(holistic_iccs)
            print(f"  Holistic ICC: {avg_holistic:.3f} (n={len(holistic_iccs)} metrics)")
        
        # Turn ICCs for this group
        turn_iccs = [r['icc'] for r in results['turn'] 
                     if r['group'] == group and r['icc'] is not None and not np.isnan(r['icc'])]
        if turn_iccs:
            avg_turn = np.mean(turn_iccs)
            print(f"  Turn-level ICC: {avg_turn:.3f} (n={len(turn_iccs)} metrics)")

if __name__ == "__main__":
    results = run_icc_analysis()
    detailed_icc_report(results)
    
    print(f"\n\nAnalysis completed! The results show the inter-rater reliability using ICC.")
    print("Higher values (closer to 1.0) indicate greater agreement among raters.")