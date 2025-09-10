#!/usr/bin/env python3
"""
Example script demonstrating pose movement visualization with preprocessing steps.
Shows how the preprocessing pipeline transforms pose data and visualizes movement patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def demonstrate_preprocessing_pipeline():
    """Demonstrate the preprocessing pipeline steps."""
    
    print("=== Pose Movement Visualization Example ===")
    print()
    print("This tool visualizes how pose preprocessing affects movement data:")
    print("1. Original pose sequence")
    print("2. Interpolated missing keypoints")
    print("3. Normalized coordinates")
    print("4. Bounding box normalization")
    print("5. Smoothed trajectories")
    print()
    
    print("Output includes:")
    print("- Frame-by-frame comparison plots")
    print("- Movement trajectory analysis")
    print("- Speed and direction analysis")
    print("- Video showing preprocessing steps (if --video flag used)")
    print()


def show_expected_outputs():
    """Show what output files to expect."""
    
    print("=== Expected Output Files ===")
    print()
    
    outputs = {
        "Frame Comparisons": [
            "{video_name}_frame_{N}_comparison.png - Side-by-side original vs processed poses"
        ],
        "Movement Analysis": [
            "{video_name}_{part}_original_movement.png - Original movement patterns",
            "{video_name}_{part}_processed_movement.png - Processed movement patterns"
        ],
        "Video Output": [
            "{video_name}_preprocessing_steps.mp4 - Video showing all preprocessing steps",
            "{video_name}_preprocessing_steps.gif - Fallback animation format"
        ],
        "Summary Data": [
            "{video_name}_processing_summary.json - Processing metadata and statistics"
        ]
    }
    
    for category, files in outputs.items():
        print(f"{category}:")
        for file in files:
            print(f"  - {file}")
        print()


def show_usage_examples():
    """Show usage examples."""
    
    print("=== Usage Examples ===")
    print()
    
    examples = [
        ("Quick visualization", "python run_pose_visualization.py"),
        ("Specific sample", "python run_pose_visualization.py --sample-idx 42"),
        ("With video output", "python run_pose_visualization.py --video --fps 20"),
        ("Custom parts analysis", "python run_pose_visualization.py --parts body face"),
        ("Limited frames", "python run_pose_visualization.py --max-frames 100"),
        ("Full options", "python run_pose_visualization.py --sample-idx 10 --video --fps 25 --parts body left_hand right_hand --max-frames 200")
    ]
    
    for description, command in examples:
        print(f"{description}:")
        print(f"  {command}")
        print()


def demonstrate_movement_analysis():
    """Demonstrate movement analysis concepts."""
    
    print("=== Movement Analysis Features ===")
    print()
    
    print("For each body part, the visualization provides:")
    print()
    print("1. 2D Trajectory Plot:")
    print("   - Shows movement path in x-y space")
    print("   - Green circle = start position")
    print("   - Red square = end position")
    print("   - Line thickness indicates movement density")
    print()
    
    print("2. Temporal Coordinate Analysis:")
    print("   - X and Y coordinates over time")
    print("   - Reveals periodic patterns and trends")
    print("   - Helps identify gesture phases")
    print()
    
    print("3. Movement Speed Analysis:")
    print("   - Frame-to-frame speed calculation")
    print("   - Identifies fast/slow movement phases")
    print("   - Useful for gesture segmentation")
    print()
    
    print("4. Preprocessing Impact:")
    print("   - Compares original vs processed movements")
    print("   - Shows how normalization affects trajectories")
    print("   - Validates preprocessing effectiveness")
    print()


def create_sample_analysis():
    """Create a sample analysis plot to demonstrate output format."""
    
    print("=== Creating Sample Analysis ===")
    
    # Generate sample movement data
    t = np.linspace(0, 4*np.pi, 100)
    
    # Simulate hand movement (figure-8 pattern with noise)
    x_original = np.sin(t) + 0.1 * np.random.randn(100)
    y_original = np.sin(2*t) + 0.1 * np.random.randn(100)
    
    # Simulate preprocessing effects
    x_processed = x_original * 0.8  # Scaling effect
    y_processed = y_original * 0.8 + 0.1  # Scaling + translation
    
    # Create demonstration plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sample Movement Analysis - Hand Gesture', fontsize=16, fontweight='bold')
    
    # 1. Trajectory comparison
    ax1.plot(x_original, y_original, 'b-', linewidth=2, alpha=0.7, label='Original')
    ax1.plot(x_processed, y_processed, 'r-', linewidth=2, alpha=0.7, label='Processed')
    ax1.scatter(x_original[0], y_original[0], color='green', s=100, marker='o', zorder=5)
    ax1.scatter(x_original[-1], y_original[-1], color='blue', s=100, marker='s', zorder=5)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. X movement over time
    frames = np.arange(len(x_original))
    ax2.plot(frames, x_original, 'b-', linewidth=2, label='Original X')
    ax2.plot(frames, x_processed, 'r-', linewidth=2, label='Processed X')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('X Coordinate')
    ax2.set_title('X Movement Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Y movement over time
    ax3.plot(frames, y_original, 'b-', linewidth=2, label='Original Y')
    ax3.plot(frames, y_processed, 'r-', linewidth=2, label='Processed Y')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Y Movement Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Speed comparison
    speed_orig = np.sqrt(np.diff(x_original)**2 + np.diff(y_original)**2)
    speed_proc = np.sqrt(np.diff(x_processed)**2 + np.diff(y_processed)**2)
    ax4.plot(frames[1:], speed_orig, 'b-', linewidth=2, label='Original Speed')
    ax4.plot(frames[1:], speed_proc, 'r-', linewidth=2, label='Processed Speed')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Speed (units/frame)')
    ax4.set_title('Movement Speed Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save sample plot
    output_path = Path("sample_movement_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Sample analysis plot saved: {output_path}")
    print("This demonstrates the type of movement analysis the tool provides.")
    print()


def check_dependencies():
    """Check if required dependencies are available."""
    
    print("=== Dependency Check ===")
    print()
    
    dependencies = [
        ("av", "Video processing library"),
        ("cv2", "OpenCV for image processing"),
        ("matplotlib", "Plotting library"),
        ("seaborn", "Statistical visualization"),
        ("numpy", "Numerical computing"),
        ("yaml", "Configuration file parsing")
    ]
    
    missing = []
    for dep, desc in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} - {desc}")
        except ImportError:
            print(f"✗ {dep} - {desc} (MISSING)")
            missing.append(dep)
    
    print()
    if missing:
        print("Missing dependencies. Install with:")
        if 'av' in missing:
            print("  pip install av")
        if 'cv2' in missing:
            print("  pip install opencv-python")
        if any(dep in missing for dep in ['matplotlib', 'seaborn', 'numpy', 'yaml']):
            print("  pip install matplotlib seaborn numpy pyyaml")
        print()
        return False
    else:
        print("All dependencies are available!")
        print()
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pose Movement Visualization Example")
    parser.add_argument("--demo-pipeline", action="store_true", 
                       help="Show preprocessing pipeline demo")
    parser.add_argument("--show-outputs", action="store_true", 
                       help="Show expected output files")
    parser.add_argument("--show-usage", action="store_true", 
                       help="Show usage examples")
    parser.add_argument("--demo-analysis", action="store_true", 
                       help="Show movement analysis features")
    parser.add_argument("--sample-plot", action="store_true", 
                       help="Create sample analysis plot")
    parser.add_argument("--check-deps", action="store_true", 
                       help="Check dependencies")
    args = parser.parse_args()
    
    # If no specific flags, show everything
    show_all = not any([args.demo_pipeline, args.show_outputs, args.show_usage, 
                       args.demo_analysis, args.sample_plot, args.check_deps])
    
    if args.check_deps or show_all:
        deps_ok = check_dependencies()
        if not deps_ok and not show_all:
            exit(1)
    
    if args.demo_pipeline or show_all:
        demonstrate_preprocessing_pipeline()
    
    if args.show_outputs or show_all:
        show_expected_outputs()
    
    if args.show_usage or show_all:
        show_usage_examples()
    
    if args.demo_analysis or show_all:
        demonstrate_movement_analysis()
    
    if args.sample_plot or show_all:
        create_sample_analysis()
    
    if show_all:
        print("=== Quick Start ===")
        print("Try running: python run_pose_visualization.py")
        print("This will process a random sample and create visualizations.")
        print()