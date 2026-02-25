import csv
import re
from collections import Counter
import random

# Read dataset
with open('/home/rbalzani/medical-vqa/Thesis/preprocessing/bounding_box/mimic_ext_mapped.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

def categorize_question(question):
    """Categorize question by type and anatomical region"""
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

# Compare sequential (first 4000) vs random sampling (4000 from 50k)
random.seed(42)
sequential_sample = data[:4000]
random_sample = random.sample(data, 4000)

print("=" * 80)
print("BIAS CALCULATION METHODOLOGY")
print("=" * 80)
print()

# Analyze both samples
for sample_name, sample in [
    ('SEQUENTIAL (first 4000)', sequential_sample),
    ('RANDOM (4000 from 50k)', random_sample)
]:
    qtypes, regions = [], []
    for row in sample:
        qtype, region = categorize_question(row['question'])
        qtypes.append(qtype)
        regions.append(region)
    
    print(f"{sample_name}:")
    print(f"  Question Types:")
    type_counts = Counter(qtypes)
    for qtype in ['tubes_devices', 'anatomical', 'disease', 'technical', 'other']:
        count = type_counts[qtype]
        pct = 100 * count / len(sample)
        print(f"    {qtype:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"  Anatomical Regions:")
    region_counts = Counter(regions)
    for region in ['lung', 'cardiac', 'mediastinal', 'chest_wall', 'other']:
        count = region_counts[region]
        pct = 100 * count / len(sample)
        print(f"    {region:20s}: {count:4d} ({pct:5.1f}%)")
    print()

# Calculate bias metrics
print("=" * 80)
print("BIAS CALCULATION")
print("=" * 80)
print()

seq_qtypes = Counter(categorize_question(row['question'])[0] for row in sequential_sample)
rand_qtypes = Counter(categorize_question(row['question'])[0] for row in random_sample)

seq_regions = Counter(categorize_question(row['question'])[1] for row in sequential_sample)
rand_regions = Counter(categorize_question(row['question'])[1] for row in random_sample)

print("Question Type Bias:")
print(f"{'Category':<25} {'Sequential %':>12} {'Random %':>12} {'Bias':>15}")
print("-" * 80)

# Key examples shown in the table
for category, seq_count in [('tubes_devices', seq_qtypes['tubes_devices']),
                             ('anatomical', seq_qtypes['anatomical'])]:
    seq_pct = 100 * seq_count / 4000
    rand_pct = 100 * rand_qtypes[category] / 4000
    
    # Calculate bias: (sequential - random) / random * 100
    bias_pct = ((seq_pct - rand_pct) / rand_pct) * 100
    
    print(f"{category:<25} {seq_pct:>11.1f}% {rand_pct:>11.1f}% {bias_pct:>+14.0f}%")
    print(f"  → Formula: ({seq_pct:.1f} - {rand_pct:.1f}) / {rand_pct:.1f} × 100 = {bias_pct:+.0f}%")

print()
print("Anatomical Region Bias:")
print(f"{'Region':<25} {'Sequential %':>12} {'Random %':>12} {'Bias':>15}")
print("-" * 80)

category = 'lung'
seq_count = seq_regions[category]
seq_pct = 100 * seq_count / 4000
rand_pct = 100 * rand_regions[category] / 4000
bias_pct = ((seq_pct - rand_pct) / rand_pct) * 100

print(f"{category:<25} {seq_pct:>11.1f}% {rand_pct:>11.1f}% {bias_pct:>+14.0f}%")
print(f"  → Formula: ({seq_pct:.1f} - {rand_pct:.1f}) / {rand_pct:.1f} × 100 = {bias_pct:+.0f}%")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
Positive bias = Over-representation in sequential sample
  Example: +103% means sequential has 2.03× more tubes/devices questions

Negative bias = Under-representation in sequential sample
  Example: -26% means sequential has 0.74× fewer lung questions

This demonstrates that sequential sampling (first 4K rows) does NOT
represent the true distribution of the full dataset.
""")
