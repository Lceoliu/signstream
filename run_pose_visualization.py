#!/usr/bin/env python3
"""
Simple runner script for pose movement visualization.
Usage: python run_pose_visualization.py [--sample-idx 42] [--video]
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run pose movement visualization")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--sample-idx", type=int, default=None,
                       help="Specific sample to visualize (random if not specified)")
    parser.add_argument("--output", type=str, default="pose_movement_results",
                       help="Output directory")
    parser.add_argument("--video", action="store_true",
                       help="Create video output (requires av library)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Video frame rate")
    parser.add_argument("--max-frames", type=int, default=150,
                       help="Maximum frames to process")
    parser.add_argument("--parts", type=str, nargs="+",
                       default=["body", "left_hand", "right_hand"],
                       help="Body parts to analyze")
    args = parser.parse_args()

    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    # Import and run the visualization
    try:
        from signstream.visualization.pose_movement_visualizer import main as viz_main
        
        # Override sys.argv to pass arguments
        sys.argv = [
            'pose_movement_visualizer.py',
            '--config', args.config,
            '--output-dir', args.output,
            '--fps', str(args.fps),
            '--max-frames', str(args.max_frames),
            '--analyze-parts'] + args.parts
        
        if args.sample_idx is not None:
            sys.argv.extend(['--sample-idx', str(args.sample_idx)])
        
        if args.video:
            sys.argv.append('--create-video')
        
        print(f"Starting pose movement visualization:")
        print(f"  Config: {args.config}")
        print(f"  Output: {args.output}")
        print(f"  Sample: {args.sample_idx if args.sample_idx is not None else 'Random'}")
        print(f"  Video: {'Yes' if args.video else 'No'}")
        print(f"  Parts: {', '.join(args.parts)}")
        print(f"  Max frames: {args.max_frames}")
        print()
        
        viz_main()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install av opencv-python matplotlib seaborn")
        sys.exit(1)
    except Exception as e:
        print(f"Error running visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()