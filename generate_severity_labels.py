"""
Generate severity labels for pothole dataset based on bounding box dimensions.

Severity classes:
- 0: Minor (small potholes, area < 0.02)
- 1: Moderate (medium potholes, 0.02 <= area < 0.08) 
- 2: Severe (large potholes, area >= 0.08)

These thresholds are based on normalized box areas (width * height).
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def calculate_box_area(width, height):
    """Calculate normalized bounding box area."""
    return width * height

def assign_severity(area):
    """Assign severity class based on box area.
    
    Thresholds (normalized):
    - Minor: area < 0.02 (~14% of image in each dimension)
    - Moderate: 0.02 <= area < 0.08 (~28% of image in each dimension)
    - Severe: area >= 0.08 (>28% of image in each dimension)
    """
    if area < 0.02:
        return 0  # Minor
    elif area < 0.08:
        return 1  # Moderate
    else:
        return 2  # Severe

def process_label_file(label_path):
    """Process a single label file and return boxes with severity."""
    boxes_with_severity = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, width, height = map(float, parts)
            
            # Calculate area and assign severity
            area = calculate_box_area(width, height)
            severity = assign_severity(area)
            
            boxes_with_severity.append({
                'class_id': int(class_id),
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'area': area,
                'severity': severity
            })
    
    return boxes_with_severity

def analyze_dataset(base_path):
    """Analyze the entire dataset to see severity distribution."""
    splits = ['train', 'valid', 'test']
    all_severities = []
    all_areas = []
    
    stats = {}
    
    for split in splits:
        labels_dir = Path(base_path) / split / 'labels'
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping...")
            continue
        
        severities = []
        areas = []
        
        for label_file in labels_dir.glob('*.txt'):
            boxes = process_label_file(label_file)
            for box in boxes:
                severities.append(box['severity'])
                areas.append(box['area'])
        
        stats[split] = {
            'total_boxes': len(severities),
            'severity_counts': Counter(severities),
            'areas': areas
        }
        
        all_severities.extend(severities)
        all_areas.extend(areas)
    
    # Overall statistics
    stats['overall'] = {
        'total_boxes': len(all_severities),
        'severity_counts': Counter(all_severities),
        'areas': all_areas
    }
    
    return stats

def print_statistics(stats):
    """Print detailed statistics about severity distribution."""
    severity_names = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
    
    print("\n" + "="*60)
    print("SEVERITY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for split, data in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Total boxes: {data['total_boxes']}")
        
        if data['total_boxes'] > 0:
            print(f"  Severity breakdown:")
            for sev_id in [0, 1, 2]:
                count = data['severity_counts'].get(sev_id, 0)
                percentage = (count / data['total_boxes']) * 100
                print(f"    {severity_names[sev_id]:10} (class {sev_id}): {count:4} ({percentage:.1f}%)")
            
            if 'areas' in data and len(data['areas']) > 0:
                areas = np.array(data['areas'])
                print(f"  Area statistics:")
                print(f"    Min:    {areas.min():.4f}")
                print(f"    Max:    {areas.max():.4f}")
                print(f"    Mean:   {areas.mean():.4f}")
                print(f"    Median: {np.median(areas):.4f}")

def visualize_distribution(stats, output_path='severity_distribution.png'):
    """Create visualization of severity distribution."""
    severity_names = ['Minor', 'Moderate', 'Severe']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot for each split
    splits = ['train', 'valid', 'test']
    for idx, split in enumerate(splits):
        ax = axes[idx // 2, idx % 2]
        
        if split in stats:
            counts = [stats[split]['severity_counts'].get(i, 0) for i in range(3)]
            colors = ['green', 'orange', 'red']
            
            bars = ax.bar(severity_names, counts, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(f'{split.capitalize()} Set', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xlabel('Severity Class', fontsize=10)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    # Overall distribution
    ax = axes[1, 1]
    if 'overall' in stats:
        counts = [stats['overall']['severity_counts'].get(i, 0) for i in range(3)]
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(severity_names, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Overall Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlabel('Severity Class', fontsize=10)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

def create_severity_annotations(base_path, output_dir='severity_labels'):
    """Create new annotation files with severity information.
    
    Format: class_id x_center y_center width height severity
    """
    output_path = Path(base_path) / output_dir
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = Path(base_path) / split / 'labels'
        if not labels_dir.exists():
            continue
        
        # Create output directory
        output_split_dir = output_path / split / 'labels'
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        for label_file in labels_dir.glob('*.txt'):
            boxes = process_label_file(label_file)
            
            # Write new annotation file
            output_file = output_split_dir / label_file.name
            with open(output_file, 'w') as f:
                for box in boxes:
                    # Original YOLO format + severity class
                    line = f"{int(box['class_id'])} {box['x_center']:.6f} {box['y_center']:.6f} " \
                           f"{box['width']:.6f} {box['height']:.6f} {box['severity']}\n"
                    f.write(line)
            
            processed_count += 1
        
        print(f"Processed {processed_count} files in {split} split")
    
    print(f"\nSeverity annotations saved to: {output_path}")
    return output_path

def main():
    # Base path to dataset
    base_path = Path(__file__).parent
    
    print("Analyzing dataset and generating severity labels...")
    print(f"Dataset path: {base_path}")
    
    # Analyze dataset
    stats = analyze_dataset(base_path)
    
    # Print statistics
    print_statistics(stats)
    
    # Visualize distribution
    visualize_distribution(stats, output_path=base_path / 'severity_distribution.png')
    
    # Create severity annotations
    print("\n" + "="*60)
    print("Creating severity annotation files...")
    print("="*60)
    output_dir = create_severity_annotations(base_path)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the severity_distribution.png to verify thresholds")
    print("2. Check sample annotations in severity_labels/ directory")
    print("3. Adjust thresholds in assign_severity() if needed")
    print("4. Use these annotations for multi-task training")

if __name__ == '__main__':
    main()
