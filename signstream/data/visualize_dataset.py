from .datasets import CSLDailyDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def visualize_video(
    poses: torch.Tensor, save_path: str = "visualization.mp4", fps: int = 30
) -> None:
    """
    Visualize a sequence of poses and save as a video.

    Args:
        poses (torch.Tensor): Tensor of shape (T, J, 3) where T is the number of frames,
                             J is the number of joints, and 3 corresponds to (x, y, confidence).
        save_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    T, J, _ = poses.shape
    # Create a blank canvas
    canvas_size = 512
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (canvas_size, canvas_size))

    for t in range(T):
        frame = canvas.copy()
        for j in range(J):
            x, y, conf = poses[t, j]
            if conf > 0.1:  # Only draw if confidence is high enough
                cv2.circle(
                    frame,
                    (int(x * canvas_size), int(y * canvas_size)),
                    5,
                    (0, 0, 255),
                    -1,
                )

        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved to {save_path}")


def main():
    dataset = CSLDailyDataset(root="CSLDaily", split="train")
    sample = dataset[0]  # Get the first sample
    poses = sample["poses"]  # Assuming poses is of shape (T, J, 3)
    visualize_video(poses, save_path="sample_visualization.mp4", fps=30)
