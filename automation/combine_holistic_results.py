#!/usr/bin/env python3
"""
Script to combine holistic evaluation results from individual metric files into a single CSV per scenario.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def combine_holistic_results(input_dir: Path, output_dir: Path) -> None:
    """
    Combine holistic evaluation results from individual metric files into a single CSV per scenario.
    """
    input_json_dir = input_dir / "json"
    input_csv_dir = input_dir / "csv"
    output_csv_dir = output_dir / "combined_csv"
    
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all unique scenario IDs from the JSON files
    json_files = list(input_json_dir.glob("*.json"))
    scenario_ids = set()
    
    for file in json_files:
        # Extract scenario ID from filename (before the first dot)
        parts = file.stem.split('.')
        if len(parts) >= 1:
            scenario_id = '.'.join(parts[:-2])  # Remove metric_id and 'holistic'/'judge'
            scenario_ids.add(scenario_id)
    
    # For each scenario, collect all metric results
    for scenario_id in scenario_ids:
        print(f"Combining results for scenario: {scenario_id}")
        
        # Find all metric files for this scenario
        scenario_files = [f for f in json_files if f.stem.startswith(f"{scenario_id}.")]
        
        all_rows: List[Dict[str, Any]] = []
        
        for file in scenario_files:
            # Parse the metric from the filename
            parts = file.stem.split('.')
            if len(parts) >= 3 and parts[-2] == 'holistic' and parts[-1] == 'judge':
                metric_id = parts[-3]  # The metric ID is before 'holistic.judge'
                
                # Load the JSON file
                try:
                    data = json.loads(file.read_text(encoding='utf-8'))
                    
                    if data.get("judge_json"):
                        metrics = data["judge_json"]["holistic_evaluation"]["metrics"]
                        
                        for metric in metrics:
                            row = {
                                "scenario_id": scenario_id,
                                "judge_model": data.get("judge_model", ""),
                                "metric_id": metric["metric_id"],
                                "number_id": metric["number_id"],
                                "score": metric["score"],
                                "rationale": metric.get("rationale", "")
                            }
                            all_rows.append(row)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
        
        # Write combined CSV for this scenario
        if all_rows:
            output_file = output_csv_dir / f"{scenario_id}.holistic.combined.csv"
            fieldnames = ["scenario_id", "judge_model", "metric_id", "number_id", "score", "rationale"]
            
            with output_file.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in all_rows:
                    w.writerow(row)
            
            print(f"  ✓ Wrote combined CSV: {output_file} ({len(all_rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Combine holistic evaluation results from individual metric files")
    ap.add_argument("--input-dir", required=True, help="Input directory containing json/csv subdirectories")
    ap.add_argument("--output-dir", default="../pilot_llm_evaluation_holistic_combined", help="Output directory for combined results")
    
    args = ap.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    combine_holistic_results(input_dir, output_dir)
    

if __name__ == "__main__":
    main()