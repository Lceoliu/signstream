# SignStream-RVQ: Multi-Channel Residual Vector Quantization for Sign Language

A PyTorch implementation of multi-channel Residual Vector Quantization (RVQ) for sign language pose sequence processing, as specified in the project requirements.

## Overview

SignStream-RVQ implements Stage 1 of a real-time sign language translation system that:

- **Multi-Channel RVQ**: Discretizes pose sequences from different body parts (face, left/right hand, body, full_body) into tokens
- **EMA Updates**: Uses exponential moving averages for stable codebook learning
- **Comprehensive Metrics**: Tracks codebook utilization, perplexity, and temporal consistency
- **Export Pipeline**: Generates token sequences with optional Run-Length Encoding (RLE)
- **Visualization Tools**: Provides comprehensive analysis and debugging visualizations

## Project Structure

```
signstream/
├── configs/
│   └── default.yaml              # Default training configuration
├── data/
│   ├── datasets.py              # CSL-Daily dataset implementation
│   └── visualize_dataset.py     # Dataset visualization tools
├── models/
│   ├── rvq/
│   │   ├── encoder.py           # Multi-body-part pose encoder
│   │   ├── decoder.py           # Symmetric pose decoder
│   │   ├── quantizer.py         # RVQ with EMA updates
│   │   └── rvq_model.py         # Complete encode->quantize->decode pipeline
│   └── metrics/
│       ├── recon.py             # Reconstruction quality metrics
│       ├── temporal.py          # Temporal consistency metrics
│       └── codebook.py          # Codebook utilization analysis
├── training/
│   ├── train_rvq.py            # Main training script
│   ├── train_rvq_multiple_gpu.py # Multi-GPU DDP training
│   ├── loop_rvq.py             # Training loop implementation
│   ├── optim.py                # Optimizer and scheduler utilities
│   ├── losses.py               # Loss functions
│   └── utils_tb.py             # TensorBoard logging utilities
├── inference/
│   ├── export_tokens.py        # Token export with RLE support
│   ├── rle.py                  # Run-Length Encoding utilities
│   └── visualizer.py           # Comprehensive visualization tools
├── io/
│   └── dataloader_utils.py     # DataLoader utilities
└── tests/
    ├── test_dataset.py         # Dataset tests
    ├── test_quantizer.py       # RVQ tests
    └── test_export.py          # Export pipeline tests
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.1.0+
- CUDA (optional, for GPU training)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd signstream-rvq
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Activate conda environment** (if using the specified environment):
   ```bash
   conda activate SLR_realtime
   ```

## Quick Start

### Training

**Single GPU Training**:
```bash
# Use the provided shell script
bash train.sh

# Or run directly
python -m signstream.training.train_rvq --config signstream/configs/default.yaml
```

**Multi-GPU Training**:
```bash
# Use the provided shell script
NUM_GPUS=4 bash train_multi.sh

# Or run directly
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 \
    signstream/training/train_rvq_multiple_gpu.py --config signstream/configs/default.yaml
```

### Token Export

```bash
# Use the provided shell script
CHECKPOINT_PATH="checkpoints/best_model.pt" NUM_SAMPLES=50 bash export_tokens.sh

# Or run directly
python -m signstream.inference.export_tokens \
    --config signstream/configs/default.yaml \
    --checkpoint checkpoints/best_model.pt \
    --num-samples 50 \
    --out exports/tokens.jsonl
```

### Visualization

```bash
# Visualize a sample with reconstruction
CHECKPOINT_PATH="checkpoints/best_model.pt" bash decode_sample.sh
```

## Configuration

The system is configured through YAML files. Key configuration sections:

### Data Configuration
```yaml
data:
  root: "./CSL-Daily"                    # Dataset root directory
  fps: 25                                # Video frame rate
  chunk_len: 10                          # Frames per chunk
  normalize:
    center_parts: true                   # Center body parts
    bbox_range: 8                        # Normalize to [-8, 8] range
  body_parts:                            # Body part keypoint ranges
    face: [24, 91]
    left_hand: [92, 112]
    right_hand: [113, 133]
    body: [1, 17]
    full_body: [1, 133]
  augment:                               # Data augmentation
    mirror: true
    time_warp: true
    dropout_prob: 0.05
```

### Model Configuration
```yaml
model:
  latent_dim: 256                        # Latent space dimension
  arch: "transformer"                    # "mlp" or "transformer"
  rvq:
    levels: 3                            # RVQ levels
    codebook_size: 1024                  # Codebook size per level
    ema_decay: 0.99                      # EMA decay rate
    commitment_beta: 0.25                # Commitment loss weight
    usage_reg: 1e-3                      # Usage regularization weight
    temporal_loss_alpha: 0.05            # Temporal consistency weight
```

### Training Configuration
```yaml
training:
  epochs: 200
  batch_size: 8
  lr: 3e-4
  wd: 0.01
  amp: "bf16"                            # Mixed precision
  save_every: 5                          # Checkpoint frequency
  eval_every: 1                          # Evaluation frequency
```

## Data Format

### Input Format
The system expects pose data in COCO WholeBody format:
- **Input shape**: `[T, 133, 3]` where T is sequence length
- **Coordinates**: `[x, y, confidence]` for each keypoint
- **Body parts**: Face (68 points), hands (21 each), body (17 points)

### Output Format
Exported tokens follow this JSON Lines format:

```json
{
  "video_name": "sample_001",
  "fps": 25,
  "chunk_len": 10,
  "tokens": {
    "face": [
      {"t": 0, "F": [12, 7]},
      {"t": 1, "F": [7, 13]}
    ],
    "left_hand": [
      {"t": 0, "LH": [37]},
      {"t": 1, "LH": ["NC", 3]}
    ]
  },
  "templates": [
    "<T0><F:c12><F:c7><LH:a37><RH:b8><B:d91>",
    "<T1><F:c7><LH:NCx3><RH:b8><B:d13>"
  ],
  "rle": true
}
```

## Key Features

### 1. Multi-Channel RVQ
- **Separate codebooks** for each body part
- **Shared encoder backbone** with type embeddings
- **Residual quantization** with 2-3 levels per part

### 2. EMA-based Learning
- **Exponential moving averages** for stable codebook updates
- **Usage regularization** to prevent codebook collapse
- **Commitment loss** for encoder-quantizer alignment

### 3. Comprehensive Metrics
- **Codebook health**: Utilization rate, perplexity, entropy
- **Temporal consistency**: Smoothness across time chunks
- **Reconstruction quality**: MSE, MAE, error distributions

### 4. Export Pipeline
- **Multi-format support**: JSON, JSONL
- **Run-Length Encoding**: Compresses static sequences
- **Template generation**: LLM-ready token formats

### 5. Visualization Tools
- **Codebook utilization** heatmaps and histograms
- **Temporal pattern** analysis with PCA
- **Reconstruction quality** comparison plots
- **Token timeline** visualizations

## Training Pipeline

The training process includes:

1. **Data Loading**: CSL-Daily dataset with chunking and augmentation
2. **Multi-part Processing**: Simultaneous training on all body parts
3. **Loss Computation**:
   - Reconstruction loss (Huber/MSE)
   - VQ commitment loss
   - Usage regularization
   - Temporal consistency loss
4. **Metrics Tracking**: Comprehensive logging to TensorBoard
5. **Health Monitoring**: Codebook utilization analysis every 5 epochs

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest signstream/tests/

# Run specific test modules
python -m pytest signstream/tests/test_quantizer.py
python -m pytest signstream/tests/test_dataset.py

# Run tests manually (if pytest not available)
python signstream/tests/test_quantizer.py
```

## Dummy Data Support

For development and testing without CSL-Daily data, the system automatically creates dummy pose sequences that maintain the same data structure and processing pipeline.

## Performance Considerations

### Memory Usage
- **Batch size scaling**: Automatically adjusts for available GPU memory
- **Gradient accumulation**: Supports effective large batch training
- **Mixed precision**: bf16/fp16 support for memory efficiency

### Multi-GPU Training
- **Data parallel**: DistributedDataParallel (DDP) support
- **Automatic scaling**: Batch size and learning rate scaling
- **Synchronized metrics**: Cross-GPU metric aggregation

### Optimization
- **EMA updates**: Efficient codebook learning without backprop
- **Chunked processing**: Memory-efficient long sequence handling
- **Vectorized operations**: Optimized tensor operations throughout

## Monitoring and Debugging

### TensorBoard Logging
- **Loss curves**: All loss components tracked separately
- **Codebook metrics**: Utilization, perplexity, entropy over time
- **Sample tokens**: Regular token samples for debugging
- **Gradient monitoring**: Gradient norms and distributions

### Health Checks
- **Codebook collapse detection**: Automatic warning for low utilization
- **Training stability**: Loss divergence detection
- **Memory monitoring**: GPU memory usage tracking

## Future Extensions

This Stage 1 implementation provides hooks for:

- **Prefix-LM integration**: Token sequences ready for LLM training
- **Streaming inference**: Architecture supports real-time processing
- **Multi-modal fusion**: Framework extensible to other modalities

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Codebook collapse**: Increase usage regularization weight
3. **Training instability**: Check learning rate and mixed precision settings
4. **Data loading errors**: Verify CSL-Daily dataset structure

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{signstream-rvq,
  title={SignStream-RVQ: Multi-Channel Residual Vector Quantization for Sign Language},
  year={2024},
  howpublished={\url{https://github.com/your-repo/signstream-rvq}}
}
```

## License

[Specify license here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the test cases for usage examples