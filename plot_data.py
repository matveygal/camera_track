#!/usr/bin/env python3
"""
Plot tracking data from saved CSV files
"""
import sys
import csv
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path

def plot_csv(filename):
    """Load and plot tracking data from CSV"""
    if not Path(filename).exists():
        print(f"Error: File '{filename}' not found")
        return
    
    # Read data
    times = []
    distances = []
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['Time (s)']))
            distances.append(float(row['Distance (mm)']))
    
    if len(times) == 0:
        print("Error: No data in file")
        return
    
    # Convert to numpy arrays
    times = np.array(times)
    distances = np.array(distances)
    
    # Apply mild smoothing to reduce discrete steps while staying close to original points
    # Savitzky-Golay with small window smooths transitions without losing sharp changes
    window = min(11, len(distances) if len(distances) % 2 == 1 else len(distances) - 1)
    if window >= 5:
        distances_smooth = savgol_filter(distances, window, 2)
    else:
        distances_smooth = distances
    
    # High-resolution interpolation for smooth line rendering
    time_interp = np.linspace(times.min(), times.max(), len(times) * 50)
    distance_interp = np.interp(time_interp, times, distances_smooth)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_interp, distance_interp, 'b-', linewidth=2, label='Distance')
    plt.plot(times, distances, 'ro', markersize=2, alpha=0.5, label='Raw measurements')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Distance (mm)', fontsize=12)
    plt.title(f'Tracking Data: {Path(filename).name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Show stats
    print(f"\n=== Statistics ===")
    print(f"Duration: {times[-1]:.2f} seconds")
    print(f"Samples: {len(times)}")
    print(f"Min distance: {min(distances):.1f} mm")
    print(f"Max distance: {max(distances):.1f} mm")
    print(f"Mean distance: {sum(distances)/len(distances):.1f} mm")
    print(f"Range: {max(distances) - min(distances):.1f} mm")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_data.py <csv_file>")
        print("\nExample: python plot_data.py tracking_20260121_143025.csv")
        
        # List available CSV files
        csv_files = list(Path('.').glob('tracking_*.csv'))
        if csv_files:
            print("\nAvailable CSV files:")
            for f in sorted(csv_files):
                print(f"  {f.name}")
        sys.exit(1)
    
    plot_csv(sys.argv[1])
