"""
Export RVQ tokens for all body parts with comprehensive formatting and RLE compression.
"""

import argparse
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from signstream.data.datasets import CSLDailyDataset
from signstream.models.rvq.rvq_model import RVQModel
from .rle import rle_encode


logger = logging.getLogger(__name__)


def load_model(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> RVQModel:
    """Load RVQ model from configuration and optional checkpoint with memory management."""
    
    # Create model with new shared backbone architecture
    model = RVQModel(
        latent_dim=config["model"]["latent_dim"],
        chunk_len=config["data"]["chunk_len"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"]["commitment_beta"],
        ema_decay=config["model"]["rvq"].get("ema_decay", 0.99),
        usage_reg=config["model"]["rvq"].get("usage_reg", 1e-3),
        arch=config["model"]["arch"],
        num_layers=config["model"].get("encoder_layer", 2),
        type_embed_dim=config["model"].get("type_embed_dim", 16),
        dropout=config["model"].get("dropout", 0.1),
        temporal_aggregation=config["model"].get("temporal_aggregation", "mean"),
    )
    
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        # Load checkpoint with map_location to avoid CUDA memory issues
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Legacy format
                model.load_state_dict(checkpoint.get('model_state', checkpoint))
        finally:
            # Clean up checkpoint immediately after loading
            del checkpoint
            import gc
            gc.collect()
    
    model.eval()
    return model


def create_token_template(codes_dict: Dict[str, List[List[int]]], chunk_idx: int) -> str:
    """
    Create formatted token template for visualization.
    
    Args:
        codes_dict: Dictionary of body part codes
        chunk_idx: Current chunk index
        
    Returns:
        Formatted token string like <T0><F:c12><LH:a37><RH:b08><B:d91><FB:e45>
    """
    template_parts = [f"<T{chunk_idx}>"]
    
    # Mapping body parts to prefixes and code spaces
    part_mapping = {
        'face': ('F', 'c'),
        'left_hand': ('LH', 'a'), 
        'right_hand': ('RH', 'b'),
        'body': ('B', 'd'),
        'full_body': ('FB', 'e')
    }
    
    for part_name, codes in codes_dict.items():
        if part_name in part_mapping and chunk_idx < len(codes):
            prefix, code_space = part_mapping[part_name]
            chunk_codes = codes[chunk_idx]
            
            if isinstance(chunk_codes, list) and len(chunk_codes) > 0:
                if isinstance(chunk_codes[0], str) and chunk_codes[0] == "NC":
                    # No-change token
                    template_parts.append(f"<{prefix}:NCx{chunk_codes[1]}>")
                else:
                    # Regular codes
                    for code in chunk_codes:
                        template_parts.append(f"<{prefix}:{code_space}{code}>")
    
    return "".join(template_parts)


def export_samples(
    model: RVQModel,
    dataset: CSLDailyDataset,
    num_samples: int,
    body_parts: List[str],
    enable_rle: bool = False,
    rle_threshold: float = 0.02,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, Any]]:
    """
    Export token sequences for multiple body parts with memory management.
    
    Args:
        model: Trained RVQ model
        dataset: Dataset to sample from
        num_samples: Number of samples to export
        body_parts: List of body parts to process
        enable_rle: Whether to apply RLE compression
        rle_threshold: Threshold for considering tokens unchanged
        device: Device to run inference on
        
    Returns:
        List of exported token data
    """
    import gc
    
    model.to(device)
    model.eval()  # Ensure model is in eval mode
    results = []
    
    logger.info(f"Exporting {num_samples} samples for parts: {body_parts}")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        sample_result = {
            "video_name": sample["name"],
            "fps": dataset.fps,
            "chunk_len": dataset.chunk_len,
            "text": sample.get("text", ""),
            "gloss": sample.get("gloss", ""),
            "num_frames": sample.get("num_frames", 0),
            "num_chunks": sample.get("num_chunks", 0),
            "tokens": {},
            "templates": [],
            "rle": enable_rle,
            "meta": {"note": "ids按各自码本空间编码，不与文本词表混用"}
        }
        
        # Process each body part with memory management
        for part in body_parts:
            if part not in sample["chunks"]:
                continue
                
            x = sample["chunks"][part]  # [N, L, K, C]
            N, L, K, C = x.shape
            x_flat = x.view(N, L, K * C).to(device)
            
            with torch.no_grad():
                _, codes, _, _, _ = model(x_flat, part)
                # Immediately move to CPU and clone to break gradient graph
                codes_cpu = codes.detach().cpu().clone()
            
            # Clean up intermediate tensors
            del x_flat, codes
            
            # Convert to lists
            codes_list = codes_cpu.tolist()  # [N, levels]
            del codes_cpu  # Clean up tensor
            
            # Format tokens by chunk
            part_tokens = []
            for chunk_idx, chunk_codes in enumerate(codes_list):
                token_entry = {
                    "t": chunk_idx,
                    part.upper().replace("_", ""): chunk_codes  # F, LH, RH, B, FB
                }
                part_tokens.append(token_entry)
            
            # Apply RLE if enabled
            if enable_rle:
                part_codes = [token[part.upper().replace("_", "")] for token in part_tokens]
                compressed_codes = rle_encode(part_codes)
                
                # Reconstruct tokens with RLE
                rle_tokens = []
                for chunk_idx, codes in enumerate(compressed_codes):
                    token_entry = {
                        "t": chunk_idx,
                        part.upper().replace("_", ""): codes
                    }
                    rle_tokens.append(token_entry)
                
                part_tokens = rle_tokens
                del compressed_codes  # Clean up
            
            sample_result["tokens"][part] = part_tokens
            del codes_list, part_tokens  # Clean up
        
        # Create template visualization (first few chunks)
        max_template_chunks = min(5, sample_result.get("num_chunks", 0))
        for chunk_idx in range(max_template_chunks):
            codes_for_chunk = {}
            for part in body_parts:
                if part in sample_result["tokens"]:
                    codes_for_chunk[part] = [
                        token[part.upper().replace("_", "")]
                        for token in sample_result["tokens"][part]
                    ]
            
            if codes_for_chunk:
                template = create_token_template(codes_for_chunk, chunk_idx)
                sample_result["templates"].append(template)
            
            del codes_for_chunk  # Clean up
        
        results.append(sample_result)
        del sample, sample_result  # Clean up sample-level variables
        
        # Memory cleanup every 5 samples
        if (i + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if (i + 1) % 10 == 0:
            logger.info(f"Exported {i + 1}/{num_samples} samples")
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def create_dummy_dataset_for_export(config: Dict[str, Any], split: str) -> CSLDailyDataset:
    """Create dummy dataset for export when real data is not available."""
    from signstream.training.train_rvq import create_dummy_dataset
    return create_dummy_dataset(config)


def main() -> None:
    """Main export function with memory management."""
    import gc
    
    parser = argparse.ArgumentParser(description="Export RVQ tokens")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to export")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of samples to export")
    parser.add_argument("--body-parts", type=str, nargs="+", 
                       default=["face", "left_hand", "right_hand", "body", "full_body"],
                       help="Body parts to export tokens for")
    parser.add_argument("--out", type=str, default="exports/tokens.jsonl",
                       help="Output file path")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--format", type=str, choices=["jsonl", "json"], default="jsonl",
                       help="Output format")
    parser.add_argument("--batch-export", action="store_true",
                       help="Export in batches to reduce memory usage")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load configuration
    logger.info(f"Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataset
    try:
        dataset = CSLDailyDataset(
            root_dir=config["data"]["root"],
            split=args.split,
            chunk_len=config["data"]["chunk_len"],
            fps=config["data"]["fps"],
            # normalize=config["data"]["normalize"]["center_parts"],
            augment=False,
            body_part_indices=config["data"]["body_parts"],
            center_indices=config["data"]["center_indices"],
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load real dataset: {e}")
        dataset = create_dummy_dataset_for_export(config, args.split)

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    model = load_model(config, args.checkpoint)
    logger.info("Model loaded successfully")

    # Export samples with memory management
    try:
        samples = export_samples(
            model=model,
            dataset=dataset,
            num_samples=args.num_samples,
            body_parts=args.body_parts,
            enable_rle=config.get("export", {}).get("enable_rle", False),
            rle_threshold=config.get("export", {}).get("rle_threshold", 0.02),
            device=device,
        )
    finally:
        # Clean up model and dataset references
        del model, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results with memory management
    try:
        if args.format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, sample in enumerate(samples):
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    # Clean up processed samples
                    if i % 50 == 0:
                        gc.collect()
        else:  # json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(samples)} samples to {output_path}")

        # Print sample output for verification
        if samples:
            logger.info("Sample token template:")
            sample = samples[0]
            for i, template in enumerate(sample.get("templates", [])[:3]):
                logger.info(f"  Chunk {i}: {template}")
            del sample  # Clean up reference
    
    finally:
        # Final cleanup
        del samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
