#!/usr/bin/env python3
"""
Stratified Random Sampling for Medical VQA Datasets
====================================================
Description:
    Creates representative stratified samples from full datasets to avoid
    sampling bias while meeting computational constraints.

    Preserves distribution across:
    - Question types (tubes/devices, anatomical, disease, technical, other)
    - Anatomical regions (lung, cardiac, mediastinal, chest_wall, other)

    Generates a single stratified CSV and distributes it to all pipeline steps.

Usage:
    # Generate 4000 samples (default):
    python3 create_stratified_samples.py

    # Custom sample size:
    python3 create_stratified_samples.py --limit 2000

    # Show distribution reports:
    python3 create_stratified_samples.py --report

References:
    - Bias Analysis: ../docs/dataset_distribution_analysis.md
    - Implementation: ../docs/STRATIFIED_SAMPLING_IMPLEMENTATION.md
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random
import sys
import tempfile

# Add current directory to path to import utils
sys.path.append(str(Path(__file__).parent))
import utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_N_SAMPLES = 4000
DEFAULT_SEED = 42

# Default input source
DEFAULT_INPUT_SOURCE = "../vqa/mimic_ext_mapped.csv"

# ==============================================================================
# CATEGORIZATION LOGIC
# ==============================================================================

def categorize_question(question: str) -> Tuple[str, str]:
    """
    Categorize question by type and anatomical region.
    """
    q = question.lower()

    # Question type
    if 'tubes/lines' in q or 'devices' in q:
        qtype = 'tubes_devices'
    elif 'disease' in q or 'abnormal' in q:
        qtype = 'disease'
    elif 'anatomical' in q:
        qtype = 'anatomical'
    elif 'technical' in q:
        qtype = 'technical'
    else:
        qtype = 'other'

    # Anatomical region
    if 'lung' in q:
        region = 'lung'
    elif any(x in q for x in ['heart', 'cardiac', 'aortic']):
        region = 'cardiac'
    elif any(x in q for x in ['mediastinum', 'hilar', 'trachea']):
        region = 'mediastinal'
    elif 'chest' in q or 'hemidiaphragm' in q:
        region = 'chest_wall'
    else:
        region = 'other'

    return qtype, region


# ==============================================================================
# STRATIFIED SAMPLING
# ==============================================================================

def stratified_sample(
    data: List[Dict],
    n_samples: int,
    seed: int = 42
) -> List[Dict]:
    """
    Perform stratified random sampling based on question type and region.
    """
    random.seed(seed)

    # Group data by strata (qtype, region)
    strata = defaultdict(list)
    for row in data:
        qtype, region = categorize_question(row['question'])
        strata[(qtype, region)].append(row)

    # Calculate original distribution
    total = len(data)
    strata_proportions = {
        stratum: len(rows) / total
        for stratum, rows in strata.items()
    }

    # Calculate target counts per stratum
    target_counts = {
        stratum: max(1, int(proportion * n_samples))
        for stratum, proportion in strata_proportions.items()
    }

    # Sample from each stratum
    sampled = []
    for stratum, target_count in target_counts.items():
        stratum_data = strata[stratum]
        sample_size = min(target_count, len(stratum_data))
        sampled.extend(random.sample(stratum_data, sample_size))

    # If we're short, add random samples from largest strata
    if len(sampled) < n_samples:
        remaining = n_samples - len(sampled)
        largest_strata = sorted(
            strata.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        for stratum, rows in largest_strata:
            if remaining <= 0:
                break
            available = [r for r in rows if r not in sampled]
            add_count = min(remaining, len(available))
            if add_count > 0:
                sampled.extend(random.sample(available, add_count))
                remaining -= add_count

    # If we're over (due to rounding), trim randomly
    if len(sampled) > n_samples:
        sampled = random.sample(sampled, n_samples)

    return sampled


# ==============================================================================
# DISTRIBUTION ANALYSIS
# ==============================================================================

def analyze_distribution(data: List[Dict], label: str = "Dataset"):
    """Analyze and return distribution statistics"""
    qtypes = Counter()
    regions = Counter()
    unique_images_set = set()

    for row in data:
        qtype, region = categorize_question(row['question'])
        qtypes[qtype] += 1
        regions[region] += 1
        unique_images_set.add(row['image_path'])

    return {
        'label': label,
        'total': len(data),
        'qtypes': qtypes,
        'regions': regions,
        'unique_images': len(unique_images_set)
    }


def print_distribution_report(original_stats: Dict, sample_stats: Dict):
    """Print distribution comparison report"""
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON REPORT")
    print("=" * 70)

    print("\nQuestion Type Distribution:")
    print(f"{'Category':<20} {'Original %':>12} {'Sample %':>12} {'Diff':>10}")
    print("-" * 70)

    all_qtypes = sorted(set(
        list(original_stats['qtypes'].keys()) +
        list(sample_stats['qtypes'].keys())
    ))

    for qtype in all_qtypes:
        orig_pct = 100 * original_stats['qtypes'][qtype] / original_stats['total']
        samp_pct = 100 * sample_stats['qtypes'][qtype] / sample_stats['total']
        diff = samp_pct - orig_pct
        print(f"{qtype:<20} {orig_pct:>11.1f}% {samp_pct:>11.1f}% {diff:>+9.1f}%")

    print("\nAnatomical Region Distribution:")
    print(f"{'Region':<20} {'Original %':>12} {'Sample %':>12} {'Diff':>10}")
    print("-" * 70)

    all_regions = sorted(set(
        list(original_stats['regions'].keys()) +
        list(sample_stats['regions'].keys())
    ))

    for region in all_regions:
        orig_pct = 100 * original_stats['regions'][region] / original_stats['total']
        samp_pct = 100 * sample_stats['regions'][region] / sample_stats['total']
        diff = samp_pct - orig_pct
        print(f"{region:<20} {orig_pct:>11.1f}% {samp_pct:>11.1f}% {diff:>+9.1f}%")

    print(f"\nImage Diversity:")
    orig_img_pct = 100 * original_stats['unique_images'] / original_stats['total']
    samp_img_pct = 100 * sample_stats['unique_images'] / sample_stats['total']
    print(f"  Original: {original_stats['unique_images']:,} unique images / {original_stats['total']:,} samples ({orig_img_pct:.1f}%)")
    print(f"  Sample:   {sample_stats['unique_images']:,} unique images / {sample_stats['total']:,} samples ({samp_img_pct:.1f}%)")
    print("=" * 70)


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create stratified random samples for medical VQA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Support both --n_samples (legacy) and --limit (standardized)
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help=f'Number of samples to select (default: {DEFAULT_N_SAMPLES})'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=DEFAULT_N_SAMPLES,
        help='Legacy alias for --limit'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_SEED})'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Print distribution comparison reports'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=DEFAULT_INPUT_SOURCE,
        help=f'Path to input CSV file (default: {DEFAULT_INPUT_SOURCE})'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve limit/n_samples
    LIMIT = args.limit if args.limit is not None else args.n_samples
    INPUT_SOURCE = Path(args.input_file)
    
    # Generate dynamic filename
    # e.g. mimic_ext_mapped -> mimic_ext_stratified
    stem = INPUT_SOURCE.stem
    if "_mapped" in stem:
        base_name = stem.replace("_mapped", "_stratified")
    else:
        base_name = f"{stem}_stratified"

    OUTPUT_FILENAME = f"{base_name}_{LIMIT}_samples.csv"
    OUTPUT_CSV = Path(tempfile.gettempdir()) / OUTPUT_FILENAME

    print("=" * 70)
    print("STRATIFIED SAMPLING FOR MEDICAL VQA DATASETS")
    print("=" * 70)
    print(f"Sample Size: {LIMIT:,}")
    print(f"Random Seed: {args.seed}")
    print(f"Input Source: {INPUT_SOURCE}")
    print(f"Output File: {OUTPUT_FILENAME}")
    print("=" * 70)

    # Check input
    if not INPUT_SOURCE.exists():
        print(f"[ERROR] Input file not found: {INPUT_SOURCE}")
        sys.exit(1)

    # Read input data
    print(f"[INFO] Reading input...")
    with open(INPUT_SOURCE, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"[INFO] Total samples in dataset: {len(data):,}")

    # Perform stratified sampling
    print(f"[INFO] Creating stratified sample of {LIMIT:,} samples...")
    sample = stratified_sample(data, LIMIT, args.seed)
    print(f"[INFO] Sampled {len(sample):,} rows")

    # Write output
    print(f"[INFO] Writing to: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        if len(sample) > 0:
            writer = csv.DictWriter(f, fieldnames=sample[0].keys())
            writer.writeheader()
            writer.writerows(sample)

    # Generate report if requested
    if args.report:
        original_stats = analyze_distribution(data, "Original")
        sample_stats = analyze_distribution(sample, "Sample")
        print_distribution_report(original_stats, sample_stats)

    print(f"\n✅ Stratified sample saved locally as: {OUTPUT_CSV}")

    # Automatic Distribution
    utils.distribute_file(OUTPUT_CSV)

    print("\n✅ All stratified samples generated and distributed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
