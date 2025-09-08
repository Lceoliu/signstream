# Changelog

## 2025-09-08 — docs: add project summary and orientation

This entry summarizes the repository so future readers can grasp the project quickly.

### Overview
- Goal: Stage 1 of a real-time sign translation system — discretize multi-part pose sequences into RVQ tokens for downstream LLMs.
- Input: COCO WholeBody pose `[T, 133, 3]` → per-part chunks with 5D per keypoint `[x, y, conf, vx, vy]`.
- Output: Per-chunk, per-part token codes (multi-level RVQ), optional RLE compression, and LLM-friendly template strings.

### Key Structure (top-level modules)
- `signstream/data`:
  - `datasets.py`: `CSLDailyDataset` loads annotations + `.npy`, applies augmentation, computes velocities, chunks to `[N, L, K, 5]`. `CSLDailyDataModule.collate_fn` pads to batch.
  - `transforms.py`: body-part intervals, normalization, interpolation, velocity and skeleton features, formatting helpers, visualization utils.
- `signstream/models/rvq`:
  - `encoder.py` (`PoseEncoder`): per-part input projection → shared backbone (`mlp` or `transformer`) with type/positional embeddings and temporal aggregation (`mean`/`max`/`attention`) → `[B, D]`.
  - `quantizer.py` (`ResidualVectorQuantizer`): multi-level residual VQ with EMA codebook updates, commitment loss, usage regularization (KL→uniform), straight-through estimator.
  - `decoder.py` (`PoseDecoder`): mirrors encoder to reconstruct `[B, L, K*5]` from `[B, D]` with per-part heads.
  - `rvq_model.py` (`RVQModel`): end-to-end encode → quantize → decode per body part; returns `recon, codes, q_loss, usage_loss, z_q`.
- `signstream/models/metrics`: codebook health (utilization, perplexity, entropy) and temporal smoothness.
- `signstream/training`:
  - `train_rvq.py`: CLI; loads YAML config, sets experiment dirs, builds datasets (real or dummy), data loaders, model, optimizer/scheduler, TensorBoard; runs `RVQTrainingLoop`.
  - `loop_rvq.py`: multi-part training/validation. Flattens valid chunks to `[B*N_valid, L, K*C]`; uses weighted reconstruction loss tailored for 5D features; AMP/bf16/fp16, grad clipping, loss scaling, checkpoints, TB logs.
  - `losses.py` and `improved_losses.py`: reconstruction losses; the weighted loss separates position vs velocity to improve stability.
  - `optim.py`: Adam/AdamW and schedulers (+warmup wrapper). `utils_tb.py`: TB logging helpers. `seed.py`: reproducibility.
- `signstream/inference`:
  - `export_tokens.py`: loads model/checkpoint; exports per-part tokens; optional RLE via `rle.py`; builds template strings like `<T0><F:c12><LH:a37>...`.
- `signstream/configs/default.yaml`: single source of truth for data paths, model, RVQ, training, logging, and export.
- `signstream/tests/` + root `test_*.py`: unit/integration tests for quantizer, dataset chunking/collate, export, architecture sanity, stability, and a small training pass.

### Data Flow (train/infer)
1. Load sample → preprocess (`transforms.process_all`) → split parts + velocities.
2. Chunk per part to `[N, L, K, 5]`; batch with padding and `chunk_mask`.
3. For each part: `[B*N_valid, L, K*C]` → `encoder` → `[B, D]` → `RVQ` (codes + losses) → `decoder` → `[B, L, K*C]`.
4. Loss: weighted recon (pos vs vel), + commitment + usage regularization; temporal loss disabled by default.
5. Validate, checkpoint, log TB; optional token sampling and codebook metrics.

### Important Defaults & Stability Notes
- RVQ: `levels=3`, `codebook_size≈1k`, EMA decay `~0.99`, commitment `beta` reduced for stability, usage regularization small but >0.
- Reconstruction loss: position weight `1.0`, velocity weight `0.1` to prevent velocity dominance.
- Training: lower LR, gradient clipping, optional AMP (`fp32/fp16/bf16`), loss scaling. Temporal loss off by default due to padding/masking concerns.
- Dataset fallback: dummy dataset is auto-created if real CSL-Daily data isn’t found (useful for local/dev).

### How to Run (typical)
- Train (single GPU): `python -m signstream.training.train_rvq --config signstream/configs/default.yaml`
- Multi-GPU: `torchrun --nproc_per_node=4 signstream/training/train_rvq_multiple_gpu.py --config signstream/configs/default.yaml`
- Export tokens: `python -m signstream.inference.export_tokens --config signstream/configs/default.yaml --checkpoint <ckpt.pt> --num-samples 50 --out exports/tokens.jsonl`

### Tests (pytest)
- All: `pytest -q`
- Examples: `pytest signstream/tests/test_quantizer.py -q`, `pytest -k dataset -q`.

### Pointers
- Update `data.root` and split paths in `default.yaml` before real training.
- Outputs (checkpoints/logs/exports) are organized under `save_path/experiment_name_timestamp/`.
- Security: never commit datasets/checkpoints; keep runs reproducible via `training.seed`.

