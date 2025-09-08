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

from .transforms import *

logger = logging.getLogger(__name__)


class CSLDailyDataset(Dataset):
    """CSL-Daily Dataset for loading pose sequences and annotations."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        chunk_len: int = 10,
        fps: int = 30,
        # normalize: bool = True,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        body_part_indices: Optional[Dict] = None,
        center_indices: Optional[Dict] = None,
        max_seq_len: Optional[int] = None,
        confidence_threshold: float = 0.1,
        chunk_stride: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
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
        self.chunk_len = int(chunk_len)
        self.fps = fps
        # self.normalize = normalize
        self.augment = augment and split == "train"
        self.augment_config = augment_config or {}
        self.max_seq_len = max_seq_len
        self.confidence_threshold = confidence_threshold
        # Sliding window controls (default: no overlap)
        if chunk_stride is not None:
            self.chunk_stride = int(chunk_stride)
        elif chunk_overlap is not None:
            self.chunk_stride = max(1, int(self.chunk_len) - int(chunk_overlap))
        else:
            self.chunk_stride = int(self.chunk_len)

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

        # Apply max sequence length if specified
        if self.max_seq_len and len(poses) > self.max_seq_len:
            poses = poses[: self.max_seq_len]

        # Convert to torch tensor
        poses = torch.from_numpy(poses).float()

        # Apply augmentation if in training mode
        if self.augment:
            poses = self._augment_poses(poses)

        body_parts = process_all(poses, self.fps, self.confidence_threshold)

        # Chunk the sequences (support sliding window if stride < chunk_len)
        if getattr(self, 'chunk_stride', self.chunk_len) != self.chunk_len:
            chunked_parts = self._chunk_sequences_sliding(body_parts)
        else:
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

        # Clean up intermediate tensors to prevent memory accumulation
        del poses
        del body_parts

        return output

    def _chunk_sequences_sliding(
        self, body_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Chunk sequences with sliding window (overlap) into fixed-length segments.

        Uses window length `self.chunk_len` and stride `self.chunk_stride`.
        Pads the tail of each chunk with the last available frame (pose),
        and zeros for velocities, to reach full window length.
        """
        chunked_parts: Dict[str, torch.Tensor] = {}

        win = int(self.chunk_len)
        stride = int(self.chunk_stride)

        for part_name, part_infos in body_parts.items():
            part_poses = part_infos['pose']  # [T, K, 3]
            part_velocities = part_infos['velocity']  # [T, K, 2]
            T, K, _ = part_poses.shape

            # Determine window starts; ensure coverage of tail
            starts = list(range(0, max(T - win + 1, 1), stride))
            if not starts:
                starts = [0]
            last_start = starts[-1]
            if last_start + win < T:
                starts.append(max(0, T - win))

            pose_chunks = []
            vel_chunks = []
            for s in starts:
                e = s + win
                pose_slice = part_poses[s:min(e, T)]
                vel_slice = part_velocities[s:min(e, T)]
                cur_len = pose_slice.shape[0]

                if cur_len < win:
                    pad_t = win - cur_len
                    if cur_len > 0:
                        last_pose = pose_slice[-1:].repeat(pad_t, 1, 1)
                    else:
                        # Degenerate case: no frames
                        last_pose = torch.zeros(win, K, 3, device=part_poses.device, dtype=part_poses.dtype)
                        pose_slice = torch.zeros(0, K, 3, device=part_poses.device, dtype=part_poses.dtype)
                    pose_slice = torch.cat([pose_slice, last_pose], dim=0)

                    pad_vel = torch.zeros(pad_t, K, 2, device=part_velocities.device, dtype=part_velocities.dtype)
                    vel_slice = torch.cat([vel_slice, pad_vel], dim=0)

                pose_chunks.append(pose_slice)
                vel_chunks.append(vel_slice)

            chunked_pose = torch.stack(pose_chunks, dim=0)  # [N, L, K, 3]
            chunked_vel = torch.stack(vel_chunks, dim=0)    # [N, L, K, 2]
            chunked = torch.cat([chunked_pose, chunked_vel], dim=-1)  # [N, L, K, 5]

            chunked_parts[part_name] = chunked

        return chunked_parts

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

    def _chunk_sequences(
        self, body_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Chunk sequences into fixed-length segments."""
        chunked_parts = {}

        for part_name, part_infos in body_parts.items():
            part_poses = part_infos['pose']  # Shape: [T, N, 3]
            T = part_infos['pose'].shape[0]
            part_velocities = part_infos['velocity']  # Shape: [T, N, 2]

            # Calculate number of chunks
            num_chunks = (T + self.chunk_len - 1) // self.chunk_len

            # Pad if necessary
            pad_len = num_chunks * self.chunk_len - T
            if pad_len > 0:
                padding = part_poses[-1:].repeat(pad_len, 1, 1)
                padding_vel = torch.zeros(pad_len, part_poses.shape[1], 2)
                part_poses = torch.cat([part_poses, padding], dim=0)
                part_velocities = torch.cat([part_velocities, padding_vel], dim=0)

            # Reshape into chunks
            chunked = part_poses.reshape(num_chunks, self.chunk_len, -1, 3)
            chunked_vel = part_velocities.reshape(num_chunks, self.chunk_len, -1, 2)
            assert chunked_vel.shape == (
                num_chunks,
                self.chunk_len,
                part_poses.shape[1],
                2,
            ), f"Unexpected velocity chunk shape: {chunked_vel.shape}, expected ({num_chunks}, {self.chunk_len}, {part_poses.shape[1]}, 2)"
            # Combine pose and velocity (optional) -> [T, N, 5]
            chunked = torch.cat(
                [
                    chunked,
                    # torch.cat(
                    #     [chunked_vel, torch.zeros_like(chunked_vel[..., :1])], dim=-1
                    # ),
                    chunked_vel,
                ],
                dim=-1,
            )
            # chunked = chunked.reshape(num_chunks, self.chunk_len, -1, 5)
            assert chunked.shape == (
                num_chunks,
                self.chunk_len,
                part_poses.shape[1],
                5,
            ), f"Unexpected chunk shape: {chunked.shape}, expected ({num_chunks}, {self.chunk_len}, {part_poses.shape[1]}, 5)"
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
                # normalize=self.data_config['normalize']['center_parts'],
                augment=True,
                augment_config=self.data_config['augment'],
                body_part_indices=self.data_config['body_parts'],
                center_indices=self.data_config['center_indices'],
                max_seq_len=self.data_config.get('max_seq_len'),
                chunk_stride=self.data_config.get('chunk_stride'),
                chunk_overlap=self.data_config.get('chunk_overlap'),
            )

            # Validation dataset
            self.val_dataset = CSLDailyDataset(
                root_dir=self.data_config['root'],
                split='val',
                chunk_len=self.data_config['chunk_len'],
                fps=self.data_config['fps'],
                # normalize=self.data_config['normalize']['center_parts'],
                augment=False,
                body_part_indices=self.data_config['body_parts'],
                center_indices=self.data_config['center_indices'],
                max_seq_len=self.data_config.get('max_seq_len'),
                chunk_stride=self.data_config.get('chunk_stride'),
                chunk_overlap=self.data_config.get('chunk_overlap'),
            )

        if stage in ('test', None):
            # Test dataset
            self.test_dataset = CSLDailyDataset(
                root_dir=self.data_config['root'],
                split='test',
                chunk_len=self.data_config['chunk_len'],
                fps=self.data_config['fps'],
                # normalize=self.data_config['normalize']['center_parts'],
                augment=False,
                body_part_indices=self.data_config['body_parts'],
                center_indices=self.data_config['center_indices'],
                max_seq_len=self.data_config.get('max_seq_len'),
                chunk_stride=self.data_config.get('chunk_stride'),
                chunk_overlap=self.data_config.get('chunk_overlap'),
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
        """Custom collate function for batching with memory optimization."""
        # Find max number of chunks in batch
        max_chunks = max(sample['num_chunks'] for sample in batch)

        # Prepare batched tensors for each body part
        body_parts = ['face', 'left_hand', 'right_hand', 'body', 'full_body']
        batched_chunks = {}

        for part in body_parts:
            part_chunks = []
            for sample in batch:
                chunks = sample['chunks'][part]
                # Pad to max_chunks if necessary - use efficient padding
                if chunks.shape[0] < max_chunks:
                    pad_len = max_chunks - chunks.shape[0]
                    # Create padding tensor with same dtype/device, more efficient
                    padding = torch.zeros(
                        (pad_len,) + chunks.shape[1:], 
                        dtype=chunks.dtype, 
                        device=chunks.device
                    )
                    chunks = torch.cat([chunks, padding], dim=0)
                part_chunks.append(chunks)

            # Stack with explicit memory cleanup
            batched_chunks[part] = torch.stack(part_chunks)
            # Explicitly delete intermediate list to free memory
            del part_chunks

        # Create attention mask
        chunk_mask = torch.zeros(len(batch), max_chunks, dtype=torch.bool)
        for i, sample in enumerate(batch):
            chunk_mask[i, : sample['num_chunks']] = True

        # Create result dict with explicit tensor creation
        result = {
            'chunks': batched_chunks,
            'chunk_mask': chunk_mask,
            'names': [s['name'] for s in batch],
            'texts': [s['text'] for s in batch],
            'glosses': [s['gloss'] for s in batch],
            'num_frames': torch.tensor([s['num_frames'] for s in batch], dtype=torch.long),
            'num_chunks': torch.tensor([s['num_chunks'] for s in batch], dtype=torch.long),
        }
        
        return result
