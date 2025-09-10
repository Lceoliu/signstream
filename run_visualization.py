#!/usr/bin/env python3
"""
Simple runner script for pose reconstruction visualization.
Usage: python run_visualization.py --checkpoint path/to/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run pose reconstruction visualization")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="visualization_results",
                       help="Output directory")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--animations", action="store_true",
                       help="Save comparison animations (large files)")
    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Import and run the visualization
    try:
        from signstream.inference.visualize_reconstruction import main as viz_main
        
        # Override sys.argv to pass arguments to the visualization script
        sys.argv = [
            'visualize_reconstruction.py',
            '--config', args.config,
            '--checkpoint', args.checkpoint,
            '--output-dir', args.output,
            '--num-samples', str(args.samples)
        ]
        
        if args.animations:
            sys.argv.append('--save-animations')
        
        print(f"Starting visualization with:")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Config: {args.config}")
        print(f"  Output: {args.output}")
        print(f"  Samples: {args.samples}")
        print()
        
        viz_main()
        
    except Exception as e:
        print(f"Error running visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()