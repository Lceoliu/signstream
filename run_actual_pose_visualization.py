#!/usr/bin/env python3
"""
Runner script for the ACTUAL pose movement visualization that matches the real data pipeline.
Shows poses exactly as processed by signstream/data/datasets.py getitem method.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Visualize ACTUAL pose movement pipeline")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--sample-idx", type=int, default=None,
                       help="Specific sample to visualize (random if not specified)")
    parser.add_argument("--output", type=str, default="actual_pose_results",
                       help="Output directory")
    parser.add_argument("--video", action="store_true",
                       help="Create video output (requires av library)")
    parser.add_argument("--analyze-chunks", action="store_true",
                       help="Analyze chunked data structure")
    parser.add_argument("--fps", type=int, default=15,
                       help="Video frame rate")
    args = parser.parse_args()

    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    # Import and run the visualization
    try:
        from signstream.visualization.pose_movement_visualizer_fixed import main as viz_main
        
        # Override sys.argv to pass arguments
        sys.argv = [
            'pose_movement_visualizer_fixed.py',
            '--config', args.config,
            '--output-dir', args.output,
            '--fps', str(args.fps)
        ]
        
        if args.sample_idx is not None:
            sys.argv.extend(['--sample-idx', str(args.sample_idx)])
        
        if args.video:
            sys.argv.append('--create-video')
            
        if args.analyze_chunks:
            sys.argv.append('--analyze-chunks')
        
        print(f"Starting ACTUAL pose movement visualization:")
        print(f"  Config: {args.config}")
        print(f"  Output: {args.output}")
        print(f"  Sample: {args.sample_idx if args.sample_idx is not None else 'Random'}")
        print(f"  Video: {'Yes' if args.video else 'No'}")
        print(f"  Analyze chunks: {'Yes' if args.analyze_chunks else 'No'}")
        print()
        print("This visualization shows poses EXACTLY as processed by the actual data pipeline:")
        print("  1. Load .npy pose file")
        print("  2. process_all() from transforms.py")
        print("  3. Global bbox normalization")
        print("  4. Interpolate low confidence points") 
        print("  5. Compute velocity")
        print("  6. Split into body parts")
        print("  7. Part-wise bbox normalization")
        print("  8. Chunk into sequences")
        print()
        
        viz_main()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install av opencv-python matplotlib seaborn")
        sys.exit(1)
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()