#!/usr/bin/env python3
"""
Example script showing how to use the visualization tools.
This demonstrates the complete workflow from checkpoint loading to pose visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Example of how to use the visualization script programmatically
def run_example_visualization():
    """Run a simple example visualization."""
    
    print("=== RVQ Pose Reconstruction Visualization Example ===")
    print()
    
    # Check for available checkpoints
    checkpoint_dirs = [
        Path("checkpoints"),
        Path("experiments"),  
        Path("outputs"),
        Path(".")
    ]
    
    checkpoint_files = []
    for dir_path in checkpoint_dirs:
        if dir_path.exists():
            checkpoint_files.extend(list(dir_path.glob("**/*.pt")))
            checkpoint_files.extend(list(dir_path.glob("**/*.pth")))
    
    if not checkpoint_files:
        print("No checkpoint files found in common directories.")
        print("Please make sure you have a trained model checkpoint available.")
        print()
        print("Expected directories:")
        for dir_path in checkpoint_dirs:
            print(f"  - {dir_path}")
        print()
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for i, ckpt in enumerate(checkpoint_files):
        print(f"  {i+1}. {ckpt}")
    print()
    
    # Use the first checkpoint for the example
    checkpoint_path = checkpoint_files[0]
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Run the visualization
    try:
        from signstream.inference.visualize_reconstruction import main as viz_main
        import sys
        
        # Override sys.argv
        original_argv = sys.argv.copy()
        sys.argv = [
            'visualize_reconstruction.py',
            '--config', 'signstream/configs/default.yaml',
            '--checkpoint', str(checkpoint_path),
            '--output-dir', 'example_visualization_output',
            '--num-samples', '3',  # Small number for quick example
            '--max-frames', '50'   # Limit frames for faster processing
        ]
        
        print("Running visualization...")
        viz_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print()
        print("=== Visualization Complete! ===")
        print("Check the 'example_visualization_output' directory for results:")
        print("  - *_error_analysis.png: Reconstruction error analysis")
        print("  - *_sample_frames.png: Sample frame comparisons")
        print("  - *_tokens.json: Exported token data")
        print("  - reconstruction_summary.json: Overall summary")
        print()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed.")
        return
    except Exception as e:
        print(f"Error running visualization: {e}")
        return


def demonstrate_token_format():
    """Demonstrate the token format used by the system."""
    
    print("=== Token Format Demonstration ===")
    print()
    
    # Example token structure
    example_tokens = {
        "video_name": "example_video_001",
        "fps": 25,
        "chunk_len": 16,
        "text": "Hello world",
        "gloss": "HELLO WORLD",
        "num_frames": 120,
        "num_chunks": 8,
        "tokens": {
            "face": [
                {"t": 0, "F": [245, 156, 89]},    # Chunk 0: [level0, level1, level2]
                {"t": 1, "F": [201, 134, 67]},    # Chunk 1: [level0, level1, level2]
                {"t": 2, "F": [189, 145, 78]},    # Chunk 2: [level0, level1, level2]
            ],
            "left_hand": [
                {"t": 0, "LH": [12, 234, 156]},   # 3-level RVQ codes
                {"t": 1, "LH": [45, 198, 123]},
                {"t": 2, "LH": [67, 211, 134]},
            ],
            "right_hand": [
                {"t": 0, "RH": [89, 176, 201]},
                {"t": 1, "RH": [123, 145, 187]},
                {"t": 2, "RH": [156, 167, 198]},
            ],
            "body": [
                {"t": 0, "B": [301, 267, 189]},
                {"t": 1, "B": [278, 234, 156]},
                {"t": 2, "B": [312, 289, 201]},
            ]
        },
        "templates": [
            "<T0><F:c245><LH:a12><RH:b89><B:d301>",
            "<T1><F:c201><LH:a45><RH:b123><B:d278>",
            "<T2><F:c189><LH:a67><RH:b156><B:d312>"
        ],
        "meta": {"note": "ids按各自码本空间编码，不与文本词表混用"}
    }
    
    print("Example token structure:")
    print(json.dumps(example_tokens, indent=2, ensure_ascii=False))
    print()
    
    print("Token explanation:")
    print("- Each body part has its own token sequence")
    print("- 't' indicates the temporal chunk index")
    print("- Each token contains 3 values: [level0, level1, level2] RVQ codes")
    print("- Template format: <T{chunk}><{part}:{code_space}{code}>")
    print("  - F: Face, LH: Left Hand, RH: Right Hand, B: Body")
    print("  - Code spaces: c(face), a(left_hand), b(right_hand), d(body), e(full_body)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RVQ Visualization Example")
    parser.add_argument("--demo-tokens", action="store_true", 
                       help="Show token format demonstration")
    parser.add_argument("--run-viz", action="store_true", 
                       help="Run example visualization")
    args = parser.parse_args()
    
    if args.demo_tokens or (not args.run_viz and not args.demo_tokens):
        demonstrate_token_format()
    
    if args.run_viz or (not args.run_viz and not args.demo_tokens):
        run_example_visualization()