import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load both human and LLM evaluation data."""
    human_holistic = pd.read_csv('human_holistic.csv')
    human_turn = pd.read_csv('human_turn.csv')
    llm_holistic = pd.read_csv('llm_holistic.csv')
    llm_turn = pd.read_csv('llm_turn.csv')
    
    return human_holistic, human_turn, llm_holistic, llm_turn

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
        index_cols = 'rater_id' if 'rater_id' in filtered_df.columns else 'judge_model'
        column_cols = 'scenario_id'
    else:  # turn
        index_cols = 'rater_id' if 'rater_id' in filtered_df.columns else 'judge_model'
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

def align_human_llm_data(human_df, llm_df, rating_type='holistic'):
    """
    Align human and LLM data for comparison by matching scenario_id, turn, and metric_id.
    """
    # Create a common identifier for merging
    if rating_type == 'holistic':
        human_df['merge_key'] = human_df['scenario_id']
        llm_df['merge_key'] = llm_df['scenario_id']
    else:  # turn
        human_df['merge_key'] = human_df['scenario_id'].astype(str) + '_' + human_df['turn'].astype(str)
        llm_df['merge_key'] = llm_df['scenario_id'].astype(str) + '_' + llm_df['turn'].astype(str)
    
    # Prepare results dataframe
    results = []
    
    # Get unique combinations of group and metric
    groups = human_df['group_id'].unique()
    metrics = human_df['metric_id'].unique()
    
    for group in groups:
        for metric in metrics:
            # Filter data for this group and metric
            human_filtered = human_df[(human_df['group_id'] == group) & (human_df['metric_id'] == metric)]
            llm_filtered = llm_df[(llm_df['group_id'] == group) & (llm_df['metric_id'] == metric)]
            
            if len(human_filtered) == 0 or len(llm_filtered) == 0:
                continue
                
            # Group human scores by merge_key and calculate mean
            human_scores = human_filtered.groupby('merge_key')['score'].mean().reset_index()
            human_scores.columns = ['merge_key', 'human_score']
            
            # Get LLM scores (assuming one score per scenario/metric combination)
            llm_scores = llm_filtered[['merge_key', 'score']].drop_duplicates()
            llm_scores.columns = ['merge_key', 'llm_score']
            
            # Merge human and LLM scores
            merged = pd.merge(human_scores, llm_scores, on='merge_key', how='inner')
            
            if len(merged) > 0:
                # Calculate correlation between human and LLM scores
                corr, _ = pearsonr(merged['human_score'], merged['llm_score'])
                
                results.append({
                    'group': group,
                    'metric': metric,
                    'correlation': corr,
                    'n_pairs': len(merged),
                    'rating_type': rating_type
                })
    
    return pd.DataFrame(results)

def run_comprehensive_icc_analysis():
    """Run comprehensive ICC analysis comparing human-human reliability and human-LLM agreement."""
    print("Loading data...")
    human_holistic, human_turn, llm_holistic, llm_turn = load_data()

    groups = ['Group A', 'Group B', 'Group C']
    holistic_metrics = sorted(human_holistic['metric_id'].unique())
    turn_metrics = sorted(human_turn['metric_id'].unique())

    print(f"\nGroups: {groups}")
    print(f"Holistic metrics: {holistic_metrics}")
    print(f"Turn metrics: {turn_metrics}")

    # Analysis results
    results = {
        'human_holistic': [],
        'human_turn': [],
        'human_llm_holistic': [],
        'human_llm_turn': []
    }

    print("\n" + "="*80)
    print("COMPREHENSIVE ICC ANALYSIS: HUMAN-HUMAN RELIABILITY AND HUMAN-LLM AGREEMENT")
    print("="*80)

    # Human-Human Holistic Analysis
    print("\n1. HUMAN-HUMAN HOLISTIC LEVEL RELIABILITY ANALYSIS")
    print("-" * 60)

    for group in groups:
        print(f"\n{group}:")

        for metric in holistic_metrics:
            icc, n_items, n_raters = calculate_icc_for_group_and_metric(
                human_holistic, group, metric, 'holistic'
            )

            result = {
                'group': group,
                'metric': metric,
                'icc': icc,
                'n_items': n_items,
                'n_raters': n_raters,
                'type': 'human_holistic'
            }

            results['human_holistic'].append(result)

            if icc is not None and not np.isnan(icc):
                print(f"  {metric:<25} ICC = {icc:.3f} (items: {n_items}, raters: {n_raters})")
            else:
                print(f"  {metric:<25} ICC = N/A (insufficient data)")

    # Human-Human Turn-level Analysis
    print("\n\n2. HUMAN-HUMAN TURN LEVEL RELIABILITY ANALYSIS")
    print("-" * 60)

    for group in groups:
        print(f"\n{group}:")

        for metric in turn_metrics:
            icc, n_items, n_raters = calculate_icc_for_group_and_metric(
                human_turn, group, metric, 'turn'
            )

            result = {
                'group': group,
                'metric': metric,
                'icc': icc,
                'n_items': n_items,
                'n_raters': n_raters,
                'type': 'human_turn'
            }

            results['human_turn'].append(result)

            if icc is not None and not np.isnan(icc):
                print(f"  {metric:<25} ICC = {icc:.3f} (items: {n_items}, raters: {n_raters})")
            else:
                print(f"  {metric:<25} ICC = N/A (insufficient data)")

    # Human-LLM Holistic Agreement Analysis
    print("\n\n3. HUMAN-LLM HOLISTIC LEVEL AGREEMENT ANALYSIS")
    print("-" * 60)
    
    human_llm_holistic_corr = align_human_llm_data(human_holistic, llm_holistic, 'holistic')
    
    for _, row in human_llm_holistic_corr.iterrows():
        result = {
            'group': row['group'],
            'metric': row['metric'],
            'correlation': row['correlation'],
            'n_pairs': row['n_pairs'],
            'type': 'human_llm_holistic'
        }
        results['human_llm_holistic'].append(result)
        
        print(f"  {row['group']:<10} {row['metric']:<25} Correlation = {row['correlation']:.3f} (pairs: {row['n_pairs']})")

    # Human-LLM Turn-level Agreement Analysis
    print("\n\n4. HUMAN-LLM TURN LEVEL AGREEMENT ANALYSIS")
    print("-" * 60)
    
    human_llm_turn_corr = align_human_llm_data(human_turn, llm_turn, 'turn')
    
    for _, row in human_llm_turn_corr.iterrows():
        result = {
            'group': row['group'],
            'metric': row['metric'],
            'correlation': row['correlation'],
            'n_pairs': row['n_pairs'],
            'type': 'human_llm_turn'
        }
        results['human_llm_turn'].append(result)
        
        print(f"  {row['group']:<10} {row['metric']:<25} Correlation = {row['correlation']:.3f} (pairs: {row['n_pairs']})")

    # Summary statistics
    print("\n\n5. SUMMARY STATISTICS")
    print("-" * 40)

    for analysis_type in ['human_holistic', 'human_turn', 'human_llm_holistic', 'human_llm_turn']:
        print(f"\n{analysis_type.replace('_', ' ').upper()}:")

        if analysis_type.startswith('human_') and not analysis_type.startswith('human_llm'):
            # For human-human analyses, we look at ICC values
            valid_values = [r['icc'] for r in results[analysis_type] if r['icc'] is not None and not np.isnan(r['icc'])]
            value_label = 'ICC'
        else:
            # For human-LLM analyses, we look at correlation values
            valid_values = [r['correlation'] for r in results[analysis_type] if r['correlation'] is not None and not np.isnan(r['correlation'])]
            value_label = 'Correlation'

        if valid_values:
            avg_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)

            print(f"  Average {value_label}: {avg_val:.3f} (±{std_val:.3f})")
            print(f"  Range: {min_val:.3f} - {max_val:.3f}")
            print(f"  Total comparisons: {len(valid_values)}")
        else:
            print(f"  No valid {value_label.lower()}s to analyze")

    # Interpretation guide
    print("\n\n6. INTERPRETATION GUIDE")
    print("-" * 40)
    print("ICC/Correlation ≥ 0.75: Excellent agreement/reliability")
    print("ICC/Correlation 0.60-0.74: Good agreement/reliability") 
    print("ICC/Correlation 0.40-0.59: Fair agreement/reliability")
    print("ICC/Correlation < 0.40: Poor agreement/reliability")

    return results, human_llm_holistic_corr, human_llm_turn_corr

def detailed_comparison_report(results, human_llm_holistic_corr, human_llm_turn_corr):
    """Generate a detailed comparison report of human-human reliability vs human-LLM agreement."""
    print("\n\n" + "="*100)
    print("DETAILED COMPARISON REPORT: HUMAN-HUMAN RELIABILITY VS HUMAN-LLM AGREEMENT")
    print("="*100)

    # Compare human-human reliability vs human-LLM agreement
    print("\nHUMAN-HUMAN RELIABILITY vs HUMAN-LLM AGREEMENT COMPARISON")
    print("-" * 70)
    
    # Get average ICC/correlation values by type
    human_holistic_iccs = [r['icc'] for r in results['human_holistic'] if r['icc'] is not None and not np.isnan(r['icc'])]
    human_turn_iccs = [r['icc'] for r in results['human_turn'] if r['icc'] is not None and not np.isnan(r['icc'])]
    human_llm_holistic_corrs = [r['correlation'] for r in results['human_llm_holistic'] if r['correlation'] is not None and not np.isnan(r['correlation'])]
    human_llm_turn_corrs = [r['correlation'] for r in results['human_llm_turn'] if r['correlation'] is not None and not np.isnan(r['correlation'])]
    
    print(f"Human-Human Holistic ICC: {np.mean(human_holistic_iccs):.3f} (n={len(human_holistic_iccs)})")
    print(f"Human-LLM Holistic Correlation: {np.mean(human_llm_holistic_corrs):.3f} (n={len(human_llm_holistic_corrs)})")
    print(f"Human-Human Turn ICC: {np.mean(human_turn_iccs):.3f} (n={len(human_turn_iccs)})")
    print(f"Human-LLM Turn Correlation: {np.mean(human_llm_turn_corrs):.3f} (n={len(human_llm_turn_corrs)})")
    
    # Detailed breakdown by group and metric
    print(f"\n\nDETAILED BREAKDOWN BY GROUP AND METRIC")
    print("-" * 70)
    
    print("\nHolistic Level:")
    print(f"{'Group':<10} {'Metric':<25} {'Human-Human ICC':<18} {'Human-LLM Corr.':<18} {'Difference':<10}")
    print("-" * 90)
    
    # Group results by group and metric for comparison
    hh_holistic_dict = {(r['group'], r['metric']): r['icc'] for r in results['human_holistic'] if r['icc'] is not None and not np.isnan(r['icc'])}
    hl_holistic_dict = {(r['group'], r['metric']): r['correlation'] for r in results['human_llm_holistic'] if r['correlation'] is not None and not np.isnan(r['correlation'])}
    
    all_combinations = set(list(hh_holistic_dict.keys()) + list(hl_holistic_dict.keys()))
    
    for group, metric in sorted(all_combinations):
        hh_icc = hh_holistic_dict.get((group, metric), np.nan)
        hl_corr = hl_holistic_dict.get((group, metric), np.nan)
        
        if not np.isnan(hh_icc) and not np.isnan(hl_corr):
            diff = hl_corr - hh_icc
            print(f"{group:<10} {metric:<25} {hh_icc:<18.3f} {hl_corr:<18.3f} {diff:<10.3f}")
        elif not np.isnan(hh_icc):
            print(f"{group:<10} {metric:<25} {hh_icc:<18.3f} {'N/A':<18} {'N/A':<10}")
        elif not np.isnan(hl_corr):
            print(f"{group:<10} {metric:<25} {'N/A':<18} {hl_corr:<18.3f} {'N/A':<10}")
    
    print("\nTurn Level:")
    print(f"{'Group':<10} {'Metric':<25} {'Human-Human ICC':<18} {'Human-LLM Corr.':<18} {'Difference':<10}")
    print("-" * 90)
    
    hh_turn_dict = {(r['group'], r['metric']): r['icc'] for r in results['human_turn'] if r['icc'] is not None and not np.isnan(r['icc'])}
    hl_turn_dict = {(r['group'], r['metric']): r['correlation'] for r in results['human_llm_turn'] if r['correlation'] is not None and not np.isnan(r['correlation'])}
    
    all_combinations = set(list(hh_turn_dict.keys()) + list(hl_turn_dict.keys()))
    
    for group, metric in sorted(all_combinations):
        hh_icc = hh_turn_dict.get((group, metric), np.nan)
        hl_corr = hl_turn_dict.get((group, metric), np.nan)
        
        if not np.isnan(hh_icc) and not np.isnan(hl_corr):
            diff = hl_corr - hh_icc
            print(f"{group:<10} {metric:<25} {hh_icc:<18.3f} {hl_corr:<18.3f} {diff:<10.3f}")
        elif not np.isnan(hh_icc):
            print(f"{group:<10} {metric:<25} {hh_icc:<18.3f} {'N/A':<18} {'N/A':<10}")
        elif not np.isnan(hl_corr):
            print(f"{group:<10} {metric:<25} {'N/A':<18} {hl_corr:<18.3f} {'N/A':<10}")

    # Reliability by group
    print(f"\n\nAGREEMENT BY GROUP")
    print("-" * 60)

    for group in ['Group A', 'Group B', 'Group C']:
        print(f"\n{group}:")
        
        # Human-Human Holistic ICCs for this group
        hh_holistic_iccs = [r['icc'] for r in results['human_holistic']
                            if r['group'] == group and r['icc'] is not None and not np.isnan(r['icc'])]
        if hh_holistic_iccs:
            avg_hh_holistic = np.mean(hh_holistic_iccs)
            print(f"  Human-Human Holistic ICC: {avg_hh_holistic:.3f} (n={len(hh_holistic_iccs)} metrics)")

        # Human-LLM Holistic correlations for this group
        hl_holistic_corrs = [r['correlation'] for r in results['human_llm_holistic']
                             if r['group'] == group and r['correlation'] is not None and not np.isnan(r['correlation'])]
        if hl_holistic_corrs:
            avg_hl_holistic = np.mean(hl_holistic_corrs)
            print(f"  Human-LLM Holistic Correlation: {avg_hl_holistic:.3f} (n={len(hl_holistic_corrs)} metrics)")

        # Human-Human Turn ICCs for this group
        hh_turn_iccs = [r['icc'] for r in results['human_turn']
                        if r['group'] == group and r['icc'] is not None and not np.isnan(r['icc'])]
        if hh_turn_iccs:
            avg_hh_turn = np.mean(hh_turn_iccs)
            print(f"  Human-Human Turn ICC: {avg_hh_turn:.3f} (n={len(hh_turn_iccs)} metrics)")

        # Human-LLM Turn correlations for this group
        hl_turn_corrs = [r['correlation'] for r in results['human_llm_turn']
                         if r['group'] == group and r['correlation'] is not None and not np.isnan(r['correlation'])]
        if hl_turn_corrs:
            avg_hl_turn = np.mean(hl_turn_corrs)
            print(f"  Human-LLM Turn Correlation: {avg_hl_turn:.3f} (n={len(hl_turn_corrs)} metrics)")

if __name__ == "__main__":
    results, human_llm_holistic_corr, human_llm_turn_corr = run_comprehensive_icc_analysis()
    detailed_comparison_report(results, human_llm_holistic_corr, human_llm_turn_corr)

    print(f"\n\nAnalysis completed! This report compares human-human reliability with human-LLM agreement.")
    print("Higher values (closer to 1.0) indicate greater agreement between evaluators.")