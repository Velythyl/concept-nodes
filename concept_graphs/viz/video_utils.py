"""
Utilities for creating and copying videos from RGB and depth frames.
"""
import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from natsort import natsorted
import cv2
import numpy as np

log = logging.getLogger(__name__)


def create_video_from_frames(
    frame_dir: Path,
    output_video_path: Path,
    fps: int = 30,
    pattern: str = "*.jpg",
    is_depth: bool = False,
) -> bool:
    """
    Create a video from a directory of image frames.
    
    Args:
        frame_dir: Directory containing the image frames
        output_video_path: Path where the output video will be saved
        fps: Frames per second for the output video
        pattern: Glob pattern to match frame files (e.g., "*.jpg", "*.png")
        is_depth: If True, treat frames as depth images and normalize/colorize them
        
    Returns:
        True if video was created successfully, False otherwise
    """
    if not frame_dir.exists():
        log.warning(f"Frame directory does not exist: {frame_dir}")
        return False
    
    # Find all matching frame files
    frame_paths = natsorted(frame_dir.glob(pattern))
    
    # Try alternative extensions if no frames found
    if len(frame_paths) == 0 and pattern == "*.jpg":
        frame_paths = natsorted(frame_dir.glob("*.png"))
    elif len(frame_paths) == 0 and pattern == "*.png":
        frame_paths = natsorted(frame_dir.glob("*.jpg"))
    
    if len(frame_paths) == 0:
        log.warning(f"No frames found in {frame_dir} with pattern {pattern}")
        return False
    
    log.info(f"Creating video from {len(frame_paths)} frames in {frame_dir}")
    
    # Read first frame to get dimensions and process it
    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        log.error(f"Failed to read first frame: {frame_paths[0]}")
        return False
    
    # Process depth frames to ensure they're in proper format
    if is_depth:
        first_frame = _process_depth_frame(first_frame)
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )
    
    if not video_writer.isOpened():
        log.error(f"Failed to open video writer for {output_video_path}")
        return False
    
    # Write frames to video
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            log.warning(f"Failed to read frame: {frame_path}")
            continue
        
        # Process depth frames
        if is_depth:
            frame = _process_depth_frame(frame)
        
        video_writer.write(frame)
    
    video_writer.release()
    log.info(f"Successfully created video: {output_video_path}")
    return True


def _process_depth_frame(frame: np.ndarray) -> np.ndarray:
    """
    Process a depth frame: normalize and apply colormap.
    
    Args:
        frame: Depth frame (can be uint8, uint16, float32, or float64)
        
    Returns:
        BGR colored depth frame (uint8, 3-channel)
    """
    # Handle different depth data types
    if frame.dtype == np.uint16:
        # 16-bit depth: normalize based on actual range in image
        frame = frame.astype(np.float32)
        # Clip to reasonable range (5 meters = 5000mm)
        frame = np.clip(frame, 0, 5000)
        # Normalize to 0-255
        if frame.max() > 0:
            frame = (frame / frame.max() * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    elif frame.dtype == np.float32 or frame.dtype == np.float64:
        # Float depth: normalize based on actual range
        frame = frame.astype(np.float32)
        frame = np.clip(frame, 0, 5.0)  # Assuming meters
        # Normalize to 0-255
        if frame.max() > 0:
            frame = (frame / frame.max() * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    elif frame.dtype != np.uint8:
        # Handle any other type by converting to uint8
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure it's single channel before applying colormap
    if len(frame.shape) == 3:
        if frame.shape[2] > 1:
            # If multi-channel, take first channel
            frame = frame[:, :, 0]
        else:
            frame = frame[:, :, 0]
    
    # Apply colormap to get BGR colored image
    colored_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    
    return colored_frame


def normalize_depth_for_visualization(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """
    Normalize depth image to 0-255 range for visualization.
    
    Args:
        depth: Depth image (can be uint16, float32, or float64)
        max_depth: Maximum depth value to clip to (in meters)
        
    Returns:
        Normalized depth as uint8 in range 0-255
    """
    # Convert to float for processing
    depth_float = depth.astype(np.float32)
    
    # Clip to reasonable range
    depth_float = np.clip(depth_float, 0, max_depth * 1000)  # Assuming depth is in mm
    
    # Normalize to 0-255
    if depth_float.max() > 0:
        depth_normalized = (depth_float / depth_float.max() * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_float, dtype=np.uint8)
    
    return depth_normalized


def copy_video(source_path: Path, dest_path: Path) -> bool:
    """
    Copy a video file from source to destination.
    
    Args:
        source_path: Source video file path
        dest_path: Destination video file path
        
    Returns:
        True if copy was successful, False otherwise
    """
    if not source_path.exists():
        log.warning(f"Source video does not exist: {source_path}")
        return False
    
    try:
        shutil.copy2(source_path, dest_path)
        log.info(f"Copied video from {source_path} to {dest_path}")
        return True
    except Exception as e:
        log.error(f"Failed to copy video: {e}")
        return False


def save_scene_videos(
    scene_path: Path,
    output_dir: Path,
    fps: int = 30,
) -> None:
    """
    Save RGB and depth videos from a scene to the output directory.
    
    This function:
    1. For RGB: Copies rgb.mp4 if it exists, otherwise creates video from rgb/ frames
    2. For Depth: Creates video from raw_depth/ if exists, else depth/ if exists
    
    Args:
        scene_path: Path to the scene directory containing rgb/, depth/, raw_depth/, etc.
        output_dir: Output directory where videos will be saved
        fps: Frames per second for created videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle RGB video
    rgb_video_path = scene_path / "rgb.mp4"
    output_rgb_video = output_dir / "rgb.mp4"
    
    if rgb_video_path.exists():
        log.info("Found existing rgb.mp4, copying it...")
        copy_video(rgb_video_path, output_rgb_video)
    else:
        log.info("No rgb.mp4 found, creating video from rgb/ frames...")
        rgb_frames_dir = scene_path / "rgb"
        if rgb_frames_dir.exists():
            create_video_from_frames(
                rgb_frames_dir,
                output_rgb_video,
                fps=fps,
                pattern="*.jpg",
                is_depth=False
            )
        else:
            log.warning(f"No rgb/ directory found in {scene_path}")
    
    # Handle depth video
    output_depth_video = output_dir / "depth.mp4"
    
    # Try raw_depth first, then depth
    raw_depth_dir = scene_path / "raw_depth"
    depth_dir = scene_path / "depth"
    
    if raw_depth_dir.exists():
        log.info("Creating depth video from raw_depth/ frames...")
        create_video_from_frames(
            raw_depth_dir,
            output_depth_video,
            fps=fps,
            pattern="*.png",
            is_depth=True
        )
    elif depth_dir.exists():
        log.info("Creating depth video from depth/ frames...")
        create_video_from_frames(
            depth_dir,
            output_depth_video,
            fps=fps,
            pattern="*.png",
            is_depth=True
        )
    else:
        log.info("No depth frames found (checked raw_depth/ and depth/), skipping depth video")
