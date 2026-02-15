#!/usr/bin/env python3
"""
Aggregate benchmark results from all evaluation reports.
Extracts IoU metrics and creates a ranked summary CSV.
"""

import re
import csv
from pathlib import Path
from typing import Dict, List, Optional

def parse_evaluation_report(report_path: Path) -> Optional[Dict]:
    """Parse an evaluation report and extract key metrics."""
    if not report_path.exists():
        return None
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract metrics using regex
    metrics = {}
    
    # Extract Mean IoU
    mean_iou_match = re.search(r'Mean IoU\s+:\s+(\d+\.\d+)', content)
    if mean_iou_match:
        metrics['mean_iou'] = float(mean_iou_match.group(1))
    
    # Extract Median IoU
    median_iou_match = re.search(r'Median IoU\s+:\s+(\d+\.\d+)', content)
    if median_iou_match:
        metrics['median_iou'] = float(median_iou_match.group(1))
    
    # Extract Std IoU
    std_iou_match = re.search(r'Std IoU\s+:\s+(\d+\.\d+)', content)
    if std_iou_match:
        metrics['std_iou'] = float(std_iou_match.group(1))
    
    # Extract accuracy @ thresholds
    for threshold in [0.1, 0.25, 0.5, 0.75]:
        pattern = rf'Acc @ {threshold}\s+:\s+(\d+\.\d+)'
        match = re.search(pattern, content)
        if match:
            metrics[f'acc@{threshold}'] = float(match.group(1))
    
    # Extract counts
    total_records_match = re.search(r'Total Records\s+:\s+(\d+)', content)
    if total_records_match:
        metrics['total_records'] = int(total_records_match.group(1))
    
    evaluated_match = re.search(r'Evaluated\s+:\s+(\d+)', content)
    if evaluated_match:
        metrics['evaluated'] = int(evaluated_match.group(1))
    
    total_gold_match = re.search(r'Total Gold Boxes\s+:\s+(\d+)', content)
    if total_gold_match:
        metrics['total_gold_boxes'] = int(total_gold_match.group(1))
    
    return metrics if metrics else None

def parse_config_name(config_name: str) -> Dict:
    """Parse configuration name into components."""
    parts = {}
    
    # Strategy
    if config_name.startswith('S_question'):
        parts['strategy'] = 'question'
    elif config_name.startswith('S_regions_composite'):
        parts['strategy'] = 'regions_composite'
    elif config_name.startswith('S_regions_explode'):
        parts['strategy'] = 'regions_explode'
    else:
        parts['strategy'] = 'unknown'
    
    # Threshold
    threshold_match = re.search(r'T(\d+\.\d+)', config_name)
    if threshold_match:
        parts['threshold'] = float(threshold_match.group(1))
    
    # CRF
    if 'C_crf_on' in config_name:
        parts['crf'] = 'on'
    elif 'C_raw' in config_name:
        parts['crf'] = 'off'
    else:
        parts['crf'] = 'unknown'
    
    # Padding
    if 'P_loose' in config_name:
        parts['padding'] = 'loose'
    elif 'P_smart' in config_name:
        parts['padding'] = 'smart'
    else:
        parts['padding'] = 'unknown'
    
    # Multi-region mode (for composite/explode)
    mode_match = re.search(r'M(\d+)', config_name)
    if mode_match:
        parts['mode'] = int(mode_match.group(1))
    
    # Include mode (for composite)
    include_match = re.search(r'I(\d+)', config_name)
    if include_match:
        parts['include'] = int(include_match.group(1))
    
    return parts

def main():
    # Set up paths
    eval_results_dir = Path('evaluation/results/benchmark_v2')
    output_csv = Path('evaluation/results/benchmark_v2_summary.csv')
    
    print("=" * 60)
    print("Aggregating Benchmark Results")
    print("=" * 60)
    
    # Collect all results
    results = []
    
    for config_dir in sorted(eval_results_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        
        config_name = config_dir.name
        report_path = config_dir / 'evaluation_report.txt'
        
        metrics = parse_evaluation_report(report_path)
        if metrics is None:
            print(f"⚠️  No report found for: {config_name}")
            continue
        
        # Parse config components
        config_parts = parse_config_name(config_name)
        
        # Combine into result
        result = {
            'config_name': config_name,
            **config_parts,
            **metrics,
        }
        results.append(result)
        print(f"✅ Parsed: {config_name} (Mean IoU: {metrics.get('mean_iou', 0):.4f})")
    
    # Sort by mean IoU (descending)
    results.sort(key=lambda x: x.get('mean_iou', 0), reverse=True)
    
    # Normalize all dictionaries to have the same keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    for result in results:
        for key in all_keys:
            if key not in result:
                result[key] = None
    
    # Write CSV
    if results:
        fieldnames = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'=' * 60}")
        print(f"✅ Summary saved to: {output_csv}")
        print(f"{'=' * 60}")
        print(f"\nTop 10 Configurations by Mean IoU:")
        print("-" * 60)
        for i, result in enumerate(results[:10], 1):
            print(f"{i:2d}. {result['config_name']:50s} | IoU: {result.get('mean_iou', 0):.4f}")
        
        print(f"\n{'=' * 60}")
        print(f"Total Configurations: {len(results)}")
        print(f"Best Mean IoU: {results[0].get('mean_iou', 0):.4f}")
        print(f"Worst Mean IoU: {results[-1].get('mean_iou', 0):.4f}")
        print(f"Average Mean IoU: {sum(r.get('mean_iou', 0) for r in results) / len(results):.4f}")
        print(f"{'=' * 60}")
    else:
        print("\n❌ No results found!")

if __name__ == '__main__':
    main()
