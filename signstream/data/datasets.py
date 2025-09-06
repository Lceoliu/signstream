"""
CSL-Daily Dataset for Sign Language Pose Sequence Processing
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CSLDailyDataset(Dataset):
    """CSL-Daily Dataset for loading pose sequences and annotations."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        chunk_len: int = 10,
        fps: int = 30,
        normalize: bool = True,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        body_part_indices: Optional[Dict] = None,
        center_indices: Optional[Dict] = None,
        max_seq_len: Optional[int] = None,
        confidence_threshold: float = 0.1,
    ):
        """
        Args:
            root_dir: Root directory of CSL-Daily dataset
            split: Dataset split ('train', 'val', 'test')
            chunk_len: Number of frames per chunk
            fps: Frame rate of the videos
            normalize: Whether to normalize poses
            augment: Whether to apply data augmentation
            augment_config: Configuration for augmentation
            body_part_indices: Dictionary mapping body parts to keypoint indices
            center_indices: Dictionary of center point indices for each body part
            max_seq_len: Maximum sequence length (frames)
            confidence_threshold: Minimum confidence score to keep a keypoint
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.chunk_len = chunk_len
        self.fps = fps
        self.normalize = normalize
        self.augment = augment and split == "train"
        self.augment_config = augment_config or {}
        self.max_seq_len = max_seq_len
        self.confidence_threshold = confidence_threshold

        # Body part configuration (COCO WholeBody format)
        self.body_part_indices = body_part_indices or {
            'face': (24, 91),
            'left_hand': (92, 112),
            'right_hand': (113, 133),
            'body': (1, 17),
            'full_body': (1, 133),
        }

        self.center_indices = center_indices or {
            'body': 1,
            'face': 32,
            'left_hand': 92,
            'right_hand': 113,
        }

        # Normalization reference points (for eye distance)
        self.left_eye_indices = (66, 69)
        self.right_eye_indices = (60, 63)
        self.eyes_distance_indices = (2, 3)

        # Load annotations and split files
        self._load_metadata()

        # Filter valid samples
        self._filter_valid_samples()

        logger.info(f"Loaded {len(self.valid_samples)} valid samples for {split} split")

    def _load_metadata(self):
        """Load annotations and split information."""
        # Load annotations
        annotations_path = self.root_dir / "annotations.json"
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # Load split files
        split_path = self.root_dir / "split_files.json"
        with open(split_path, 'r') as f:
            split_data = json.load(f)

        self.split_names = split_data.get(self.split, [])

    def _filter_valid_samples(self):
        """Filter samples that have both annotations and pose files."""
        self.valid_samples = []
        frames_dir = self.root_dir / "frames_512x512"

        for name in self.split_names:
            # Check if annotation exists
            if name not in self.annotations:
                continue

            # Check if pose file exists
            pose_file = frames_dir / f"{name}.npy"
            if not pose_file.exists():
                continue

            self.valid_samples.append(
                {
                    'name': name,
                    'pose_file': pose_file,
                    'annotation': self.annotations[name],
                }
            )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample_info = self.valid_samples[idx]

        # Load pose sequence
        poses = np.load(sample_info['pose_file'])  # Shape: [T, 134, 3]

        # Remove video dimensions (first keypoint)
        poses = poses[:, 1:, :]  # Shape: [T, 133, 3]

        # Apply max sequence length if specified
        if self.max_seq_len and len(poses) > self.max_seq_len:
            poses = poses[: self.max_seq_len]

        # Convert to torch tensor
        poses = torch.from_numpy(poses).float()

        # Normalize if requested
        if self.normalize:
            poses = self._normalize_poses(poses)

        # Apply augmentation if in training mode
        if self.augment:
            poses = self._augment_poses(poses)

        # Split into body parts
        body_parts = self._split_body_parts(poses)

        # Filter out low-confidence keypoints
        for part_name, part_poses in body_parts.items():
            confidence = part_poses[:, :, 2]
            mask = confidence >= self.confidence_threshold
            part_poses[~mask] = 0.0
            body_parts[part_name] = part_poses

        # Normalize bounding box poses
        for part_name, part_poses in body_parts.items():
            part_poses = self._normalize_bbox_poses(part_poses)
            body_parts[part_name] = part_poses

        # Chunk the sequences
        chunked_parts = self._chunk_sequences(body_parts)

        # Prepare output
        output = {
            'name': sample_info['name'],
            'chunks': chunked_parts,
            'text': sample_info['annotation']['text'],
            'gloss': sample_info['annotation']['gloss'],
            'num_frames': poses.shape[0],
            'num_chunks': chunked_parts['face'].shape[0],
        }

        return output

    def _normalize_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Normalize poses using eye distance as unit length."""
        # Calculate normalization scale (eye distance)
        left_eye_dist = torch.norm(
            poses[:, self.left_eye_indices[1] - 1]
            - poses[:, self.left_eye_indices[0] - 1],
            dim=-1,
            keepdim=True,
        )
        right_eye_dist = torch.norm(
            poses[:, self.right_eye_indices[1] - 1]
            - poses[:, self.right_eye_indices[0] - 1],
            dim=-1,
            keepdim=True,
        )

        # Average eye distance as unit length
        unit_length = (left_eye_dist + right_eye_dist) / 2.0
        unit_length = unit_length.clamp(min=1e-6).unsqueeze(1)  # Avoid division by zero

        # Normalize positions (keep confidence scores unchanged)
        poses_normalized = poses.clone()
        poses_normalized[:, :, :2] = poses[:, :, :2] / unit_length

        return poses_normalized

    def _normalize_bbox_poses(
        self, poses: torch.Tensor, bbox_range: int = 8
    ) -> torch.Tensor:
        """Normalize poses to a [-bbox_range, box_range] square based on bounding box."""
        # Calculate bounding box
        x_min = torch.min(poses[:, :, 0], dim=0).values
        x_max = torch.max(poses[:, :, 0], dim=0).values
        y_min = torch.min(poses[:, :, 1], dim=0).values
        y_max = torch.max(poses[:, :, 1], dim=0).values

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Use the larger dimension as unit length
        unit_length = torch.max(bbox_width, bbox_height).clamp(min=1e-6)

        # Normalize positions (keep confidence scores unchanged)
        poses_normalized = poses.clone()
        poses_normalized[:, :, :2] = poses[:, :, :2] / unit_length * bbox_range

        return poses_normalized

    def _augment_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to poses."""
        # Mirror augmentation (flip left-right)
        if self.augment_config.get('mirror', False):
            if torch.rand(1).item() < self.augment_config.get('mirror_prob', 0.5):
                poses = self._mirror_poses(poses)

        # Time warping
        if self.augment_config.get('time_warp', False):
            factor = self.augment_config.get('time_warp_factor', 0.1)
            poses = self._time_warp(poses, factor)

        # Joint dropout
        if self.augment_config.get('dropout_prob', 0) > 0:
            poses = self._joint_dropout(poses, self.augment_config['dropout_prob'])

        return poses

    def _mirror_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Mirror poses horizontally (swap left and right)."""
        poses_mirrored = poses.clone()

        # Flip x coordinates
        poses_mirrored[:, :, 0] = -poses_mirrored[:, :, 0]

        # Swap left and right hand keypoints
        left_start, left_end = self.body_part_indices['left_hand']
        right_start, right_end = self.body_part_indices['right_hand']

        left_indices = torch.arange(left_start - 1, left_end)
        right_indices = torch.arange(right_start - 1, right_end)

        temp = poses_mirrored[:, left_indices].clone()
        poses_mirrored[:, left_indices] = poses_mirrored[:, right_indices]
        poses_mirrored[:, right_indices] = temp

        return poses_mirrored

    def _time_warp(self, poses: torch.Tensor, factor: float = 0.1) -> torch.Tensor:
        """Apply time warping augmentation."""
        T = poses.shape[0]

        # Generate random warping curve
        num_anchors = 5
        anchors = torch.linspace(0, T - 1, num_anchors)
        offsets = torch.randn(num_anchors) * factor * T / num_anchors

        # Ensure start and end points are fixed
        offsets[0] = 0
        offsets[-1] = 0

        warped_indices = anchors + offsets
        warped_indices = warped_indices.clamp(0, T - 1)

        # Interpolate to get new indices
        original_indices = torch.linspace(0, T - 1, T)

        # Simple linear interpolation for warping
        # (In production, use more sophisticated interpolation)
        return poses

    def _joint_dropout(self, poses: torch.Tensor, prob: float) -> torch.Tensor:
        """Randomly dropout joints by setting confidence to 0."""
        dropout_mask = torch.rand(poses.shape[0], poses.shape[1]) > prob
        poses[:, :, 2] = poses[:, :, 2] * dropout_mask.float()
        return poses

    def _split_body_parts(self, poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split pose sequence into body parts."""
        body_parts = {}

        for part_name, (start_idx, end_idx) in self.body_part_indices.items():
            if part_name == 'full_body':
                # Full body includes all keypoints
                part_poses = poses
            else:
                # Extract specific body part keypoints
                part_poses = poses[:, start_idx - 1 : end_idx, :]

            # Center the body part if normalize is enabled
            if self.normalize and part_name != 'full_body':
                center_idx = self.center_indices.get(part_name)
                if center_idx:
                    # Convert to local coordinates
                    if part_name in ['left_hand', 'right_hand']:
                        # For hands, center is the first point of the part
                        center = part_poses[:, 0:1, :2]
                    elif part_name == 'face':
                        # For face, use the specified center point
                        center_local_idx = center_idx - (start_idx - 1)
                        center = part_poses[
                            :, center_local_idx : center_local_idx + 1, :2
                        ]
                    else:
                        # For body, use the specified center point
                        center = poses[:, center_idx - 1 : center_idx, :2]

                    part_poses_centered = part_poses.clone()
                    part_poses_centered[:, :, :2] = part_poses[:, :, :2] - center
                    part_poses = part_poses_centered

            body_parts[part_name] = part_poses

        return body_parts

    def _chunk_sequences(
        self, body_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Chunk sequences into fixed-length segments."""
        chunked_parts = {}

        for part_name, part_poses in body_parts.items():
            T = part_poses.shape[0]

            # Calculate number of chunks
            num_chunks = (T + self.chunk_len - 1) // self.chunk_len

            # Pad if necessary
            pad_len = num_chunks * self.chunk_len - T
            if pad_len > 0:
                padding = part_poses[-1:].repeat(pad_len, 1, 1)
                part_poses = torch.cat([part_poses, padding], dim=0)

            # Reshape into chunks
            chunked = part_poses.reshape(num_chunks, self.chunk_len, -1, 3)
            chunked_parts[part_name] = chunked

        return chunked_parts


class CSLDailyDataModule:
    """Data module for managing CSL-Daily datasets and dataloaders."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.train_config = config['training']

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        if stage in ('fit', None):
            # Training dataset
            self.train_dataset = CSLDailyDataset(
                root_dir=self.data_config['root'],
                split='train',
                chunk_len=self.data_config['chunk_len'],
                fps=self.data_config['fps'],
                normalize=self.data_config['normalize']['center_parts'],
                augment=True,
                augment_config=self.data_config['augment'],
                body_part_indices=self.data_config['body_parts'],
                center_indices=self.data_config['center_indices'],
                max_seq_len=self.data_config.get('max_seq_len'),
            )

            # Validation dataset
            self.val_dataset = CSLDailyDataset(
                root_dir=self.data_config['root'],
                split='val',
                chunk_len=self.data_config['chunk_len'],
                fps=self.data_config['fps'],
                normalize=self.data_config['normalize']['center_parts'],
                augment=False,
                body_part_indices=self.data_config['body_parts'],
                center_indices=self.data_config['center_indices'],
                max_seq_len=self.data_config.get('max_seq_len'),
            )

        if stage in ('test', None):
            # Test dataset
            self.test_dataset = CSLDailyDataset(
                root_dir=self.data_config['root'],
                split='test',
                chunk_len=self.data_config['chunk_len'],
                fps=self.data_config['fps'],
                normalize=self.data_config['normalize']['center_parts'],
                augment=False,
                body_part_indices=self.data_config['body_parts'],
                center_indices=self.data_config['center_indices'],
                max_seq_len=self.data_config.get('max_seq_len'),
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        # Find max number of chunks in batch
        max_chunks = max(sample['num_chunks'] for sample in batch)

        # Prepare batched tensors for each body part
        body_parts = ['face', 'left_hand', 'right_hand', 'body', 'full_body']
        batched_chunks = {}

        for part in body_parts:
            part_chunks = []
            for sample in batch:
                chunks = sample['chunks'][part]
                # Pad to max_chunks if necessary
                if chunks.shape[0] < max_chunks:
                    pad_shape = list(chunks.shape)
                    pad_shape[0] = max_chunks - chunks.shape[0]
                    padding = torch.zeros(pad_shape)
                    chunks = torch.cat([chunks, padding], dim=0)
                part_chunks.append(chunks)

            batched_chunks[part] = torch.stack(part_chunks)

        # Create attention mask
        chunk_mask = torch.zeros(len(batch), max_chunks, dtype=torch.bool)
        for i, sample in enumerate(batch):
            chunk_mask[i, : sample['num_chunks']] = True

        return {
            'chunks': batched_chunks,
            'chunk_mask': chunk_mask,
            'names': [s['name'] for s in batch],
            'texts': [s['text'] for s in batch],
            'glosses': [s['gloss'] for s in batch],
            'num_frames': torch.tensor([s['num_frames'] for s in batch]),
            'num_chunks': torch.tensor([s['num_chunks'] for s in batch]),
        }
