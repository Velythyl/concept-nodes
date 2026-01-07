"""
Utilities for creating and copying videos from RGB and depth frames.
"""
import os
import concurrent.futures
import shutil
import logging
import subprocess
import tempfile
import json
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from natsort import natsorted
import cv2
import numpy as np
from tqdm import trange

log = logging.getLogger(__name__)


def get_video_properties(video_path: Path) -> Optional[Dict[str, float]]:
    """
    Get video properties (duration, fps, frame count) using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with 'duration', 'fps', and 'frame_count' keys, or None if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=duration,r_frame_rate,nb_read_packets',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        
        # Parse frame rate (it's in format "30/1" or "30000/1001")
        fps_parts = stream['r_frame_rate'].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        duration = float(stream.get('duration', 0))
        frame_count = int(stream.get('nb_read_packets', 0))
        
        # If duration is not available, calculate it from frame count and fps
        if duration == 0 and frame_count > 0 and fps > 0:
            duration = frame_count / fps
        
        return {
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        log.error(f"Failed to get video properties: {e}")
        return None
    except FileNotFoundError:
        log.error("ffprobe not found. Please install ffmpeg/ffprobe.")
        return None


def get_video_dimensions(video_path: Path) -> Optional[Tuple[int, int]]:
    """Return (width, height) for the first video stream via ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        log.error(f"Failed to get video dimensions: {e}")
        return None
    except FileNotFoundError:
        log.error("ffprobe not found. Please install ffmpeg/ffprobe.")
        return None


def _convert_to_web_compatible(input_path: Path, output_path: Path) -> bool:
    """
    Convert a video to web-compatible H.264 format using ffmpeg.
    
    Args:
        input_path: Path to the input video file
        output_path: Path where the web-compatible video will be saved
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Use ffmpeg to convert to H.264 with web-compatible settings
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if exists
            '-i', str(input_path),
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'fast',  # Encoding speed/quality tradeoff
            '-crf', '23',  # Quality (lower = better, 23 is default)
            '-pix_fmt', 'yuv420p',  # Pixel format for web compatibility
            '-movflags', '+faststart',  # Enable fast start for web streaming
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        log.info(f"Successfully converted video to web-compatible format: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"ffmpeg conversion failed: {e.stderr}")
        return False
    except FileNotFoundError:
        log.error("ffmpeg not found. Please install ffmpeg for web-compatible video output.")
        return False


def create_video_from_frames(
    frame_dir: Path,
    output_video_path: Path,
    fps: int = 30,
    pattern: str = "*.jpg",
    is_depth: bool = False,
    web_compatible: bool = True,
) -> bool:
    """
    Create a video from a directory of image frames.
    
    Args:
        frame_dir: Directory containing the image frames
        output_video_path: Path where the output video will be saved
        fps: Frames per second for the output video
        pattern: Glob pattern to match frame files (e.g., "*.jpg", "*.png")
        is_depth: If True, treat frames as depth images and normalize/colorize them
        web_compatible: If True, use ffmpeg to convert to H.264 for web compatibility
        
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
    
    # Determine output path - use temp file if we need to convert
    if web_compatible:
        temp_video_path = output_video_path.parent / f".temp_{output_video_path.name}"
        write_path = temp_video_path
    else:
        write_path = output_video_path
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(write_path),
        fourcc,
        fps,
        (width, height)
    )
    
    if not video_writer.isOpened():
        log.error(f"Failed to open video writer for {write_path}")
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
    
    # Convert to web-compatible format if requested
    if web_compatible:
        success = _convert_to_web_compatible(temp_video_path, output_video_path)
        # Clean up temp file
        if temp_video_path.exists():
            temp_video_path.unlink()
        if not success:
            log.warning("Failed to convert to web-compatible format, falling back to mp4v")
            # Fall back: re-create without conversion
            return create_video_from_frames(
                frame_dir, output_video_path, fps, pattern, is_depth, web_compatible=False
            )
    
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


def copy_video(source_path: Path, dest_path: Path, ensure_web_compatible: bool = True) -> bool:
    """
    Copy a video file from source to destination.
    
    Args:
        source_path: Source video file path
        dest_path: Destination video file path
        ensure_web_compatible: If True, re-encode to H.264 for web compatibility
        
    Returns:
        True if copy was successful, False otherwise
    """
    if not source_path.exists():
        log.warning(f"Source video does not exist: {source_path}")
        return False
    
    try:
        if ensure_web_compatible:
            # Re-encode to ensure web compatibility
            success = _convert_to_web_compatible(source_path, dest_path)
            if not success:
                log.warning("Failed to convert, falling back to simple copy")
                shutil.copy2(source_path, dest_path)
        else:
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
    3. Ensures depth video duration matches RGB video duration
    
    All three operations (RGB, depth, RGBD) are parallelized using threads for efficiency.
    
    Args:
        scene_path: Path to the scene directory containing rgb/, depth/, raw_depth/, etc.
        output_dir: Output directory where videos will be saved
        fps: Frames per second for created videos (used when RGB is created from frames)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define worker functions for each parallel task
    def _save_rgb_video():
        """Worker function to save RGB video."""
        rgb_video_path = scene_path / "rgb.mp4"
        output_rgb_video = output_dir / "rgb.mp4"
        
        if output_rgb_video.exists():
            log.info(f"Reusing existing RGB video at {output_rgb_video}")
        elif rgb_video_path.exists():
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
    
    def _save_depth_video():
        """Worker function to save depth video."""
        output_rgb_video = output_dir / "rgb.mp4"
        output_depth_video = output_dir / "depth.mp4"
        
        # Get RGB video properties to match depth duration
        rgb_props = None
        if output_rgb_video.exists():
            rgb_props = get_video_properties(output_rgb_video)
            if rgb_props:
                log.info(f"RGB video: duration={rgb_props['duration']:.2f}s, fps={rgb_props['fps']:.2f}, frames={rgb_props['frame_count']}")
        elif (scene_path / "rgb.mp4").exists():
            # Fallback: use scene's rgb.mp4 if output didn't create/copy it
            rgb_props = get_video_properties(scene_path / "rgb.mp4")
            if rgb_props:
                log.info(f"Using scene RGB video properties: duration={rgb_props['duration']:.2f}s, fps={rgb_props['fps']:.2f}, frames={rgb_props['frame_count']}")
        
        # Try raw_depth first, then depth
        raw_depth_dir = scene_path / "raw_depth"
        depth_dir = scene_path / "depth"
        
        # Determine which depth directory to use and count frames
        depth_frames_dir = None
        depth_pattern = "*.png"
        
        if raw_depth_dir.exists():
            depth_frames_dir = raw_depth_dir
            log.info("Creating depth video from raw_depth/ frames...")
        elif depth_dir.exists():
            depth_frames_dir = depth_dir
            log.info("Creating depth video from depth/ frames...")
        
        if output_depth_video.exists():
            log.info(f"Reusing existing depth video at {output_depth_video}")
        elif depth_frames_dir:
            # Calculate appropriate fps for depth video to match RGB duration
            depth_fps = fps  # Default
            
            if rgb_props and rgb_props['duration'] > 0:
                # Count depth frames
                depth_frame_paths = natsorted(depth_frames_dir.glob(depth_pattern))
                if len(depth_frame_paths) == 0:
                    depth_frame_paths = natsorted(depth_frames_dir.glob("*.jpg"))
                
                num_depth_frames = len(depth_frame_paths)
                
                if num_depth_frames > 0:
                    # Calculate fps to match RGB duration
                    depth_fps = num_depth_frames / rgb_props['duration']
                    log.info(f"Adjusting depth video fps to {depth_fps:.2f} to match RGB duration ({num_depth_frames} frames / {rgb_props['duration']:.2f}s)")
            
            create_video_from_frames(
                depth_frames_dir,
                output_depth_video,
                fps=depth_fps,
                pattern=depth_pattern,
                is_depth=True
            )
            
            # Verify depth video duration
            if output_depth_video.exists():
                depth_props = get_video_properties(output_depth_video)
                if depth_props:
                    log.info(f"Depth video: duration={depth_props['duration']:.2f}s, fps={depth_props['fps']:.2f}, frames={depth_props['frame_count']}")
        else:
            log.info("No depth frames found (checked raw_depth/ and depth/), skipping depth video")
    
    def _save_rgbd_video():
        """Worker function to save side-by-side RGBD video."""
        try:
            sxs_success = create_side_by_side_rgbd_video(
                scene_path=scene_path,
                output_dir=output_dir,
                fps=fps,
            )
            if sxs_success:
                log.info(f"Created side-by-side RGBD video at {output_dir / 'rgbd.mp4'}")
            else:
                log.warning("Failed to create side-by-side RGBD video.")
        except Exception as e:
            log.error(f"Exception while creating side-by-side RGBD video: {e}")
    
    # Run all three operations in parallel using threads
    log.info("Starting parallel video creation (RGB, depth, RGBD)...")
    
    rgb_thread = threading.Thread(target=_save_rgb_video, name="RGB-Video")
    depth_thread = threading.Thread(target=_save_depth_video, name="Depth-Video")
    rgbd_thread = threading.Thread(target=_save_rgbd_video, name="RGBD-Video")
    
    # Start all threads
    rgb_thread.start()
    depth_thread.start()
    rgbd_thread.start()
    
    # Wait for all threads to complete
    rgb_thread.join()
    depth_thread.join()
    rgbd_thread.join()
    
    log.info("All video creation tasks completed.")


def create_side_by_side_rgbd_video(
    scene_path: Path,
    output_dir: Path,
    fps: int = 30,
    max_workers: Optional[int] = None,
) -> bool:
    """
    Create a side-by-side video combining RGB (left) and depth (right) frames.
    
    This function:
    1. Creates a temporary directory
    2. Copies RGB frames from rgb/ directory
    3. Copies depth frames from raw_depth/ or depth/ directory
    4. Creates concatenated side-by-side images (parallelized with futures)
    5. Generates rgbd.mp4 video with same duration as rgb.mp4
    6. Saves the result in the output directory
    
    Args:
        scene_path: Path to the scene directory containing rgb/, depth/, raw_depth/, etc.
        output_dir: Output directory where videos will be saved (same as rgb.mp4/depth.mp4)
        fps: Frames per second for the output video
        max_workers: Maximum number of threads to use for parallel frame concatenation
        
    Returns:
        True if video was created successfully, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RGB video properties to match duration
    output_rgb_video = output_dir / "rgb.mp4"
    rgb_props = None
    if output_rgb_video.exists():
        rgb_props = get_video_properties(output_rgb_video)
        if rgb_props:
            log.info(f"RGB video: duration={rgb_props['duration']:.2f}s, fps={rgb_props['fps']:.2f}, frames={rgb_props['frame_count']}")
    elif (scene_path / "rgb.mp4").exists():
        # Fallback to scene RGB if output copy isn't present
        rgb_props = get_video_properties(scene_path / "rgb.mp4")
        if rgb_props:
            log.info(f"Using scene RGB video properties: duration={rgb_props['duration']:.2f}s, fps={rgb_props['fps']:.2f}, frames={rgb_props['frame_count']}")
    
    # Locate RGB and depth frame directories
    rgb_frames_dir = scene_path / "rgb"
    raw_depth_dir = scene_path / "raw_depth"
    depth_dir = scene_path / "depth"
    
    # Determine depth directory to use
    depth_frames_dir = None
    if raw_depth_dir.exists():
        depth_frames_dir = raw_depth_dir
        log.info("Using raw_depth/ for depth frames")
    elif depth_dir.exists():
        depth_frames_dir = depth_dir
        log.info("Using depth/ for depth frames")
    
    if not rgb_frames_dir.exists():
        log.error(f"RGB frames directory not found: {rgb_frames_dir}")
        return False
    
    if depth_frames_dir is None:
        log.error("No depth frames directory found (checked raw_depth/ and depth/)")
        return False
    
    # Get sorted lists of frames
    rgb_frame_paths = natsorted(list(rgb_frames_dir.glob("*.jpg")))
    if len(rgb_frame_paths) == 0:
        rgb_frame_paths = natsorted(list(rgb_frames_dir.glob("*.png")))
    
    depth_frame_paths = natsorted(list(depth_frames_dir.glob("*.png")))
    if len(depth_frame_paths) == 0:
        depth_frame_paths = natsorted(list(depth_frames_dir.glob("*.jpg")))
    
    if len(rgb_frame_paths) == 0:
        log.error(f"No RGB frames found in {rgb_frames_dir}")
        return False
    
    if len(depth_frame_paths) == 0:
        log.error(f"No depth frames found in {depth_frames_dir}")
        return False
    
    log.info(f"Found {len(rgb_frame_paths)} RGB frames and {len(depth_frame_paths)} depth frames")
    
    # Create temporary directory for concatenated frames
    temp_dir = Path(tempfile.mkdtemp(prefix="rgbd_concat_", dir="/tmp"))
    log.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Determine number of frames to process (minimum of both)
        num_frames = min(len(rgb_frame_paths), len(depth_frame_paths))
        
        # Load first frames to get dimensions
        first_rgb = cv2.imread(str(rgb_frame_paths[0]))
        first_depth = cv2.imread(str(depth_frame_paths[0]), cv2.IMREAD_UNCHANGED)
        
        if first_rgb is None or first_depth is None:
            log.error("Failed to load first frames")
            return False
        
        # Process depth frame to get colored version
        first_depth_colored = _process_depth_frame(first_depth)
        
        rgb_height, rgb_width = first_rgb.shape[:2]
        depth_height, depth_width = first_depth_colored.shape[:2]
        
        # Resize depth to match RGB height if needed
        if depth_height != rgb_height:
            aspect_ratio = depth_width / depth_height
            new_width = int(rgb_height * aspect_ratio)
            first_depth_colored = cv2.resize(first_depth_colored, (new_width, rgb_height))
            depth_height, depth_width = first_depth_colored.shape[:2]
            log.info(f"Resized depth frames to match RGB height: {depth_width}x{depth_height}")
        
        log.info(f"RGB dimensions: {rgb_width}x{rgb_height}, Depth dimensions: {depth_width}x{depth_height}")
        
        # Create concatenated frames in parallel
        if max_workers is None:
            # Use a sensible default; OpenCV releases GIL for many ops
            cpu_count = os.cpu_count() or 4
            max_workers = min(32, max(4, cpu_count))
        log.info(f"Creating {num_frames} concatenated frames in parallel (workers={max_workers})...")

        def _concat_and_write(i: int) -> bool:
            try:
                rgb_frame = cv2.imread(str(rgb_frame_paths[i]))
                depth_frame = cv2.imread(str(depth_frame_paths[i]), cv2.IMREAD_UNCHANGED)
                if rgb_frame is None or depth_frame is None:
                    log.warning(f"Failed to load frame {i}, skipping")
                    return False
                depth_colored = _process_depth_frame(depth_frame)
                if depth_colored.shape[0] != rgb_height:
                    aspect_ratio = depth_colored.shape[1] / depth_colored.shape[0]
                    new_width = int(rgb_height * aspect_ratio)
                    depth_colored = cv2.resize(depth_colored, (new_width, rgb_height))
                concatenated = np.hstack([rgb_frame, depth_colored])
                output_frame_path = temp_dir / f"frame_{i:06d}.png"
                ok = cv2.imwrite(str(output_frame_path), concatenated)
                if not ok:
                    log.warning(f"Failed to write concatenated frame {i}")
                return ok
            except Exception as e:
                log.warning(f"Error processing frame {i}: {e}")
                return False

        successes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_concat_and_write, i) for i in range(num_frames)]
            for f in concurrent.futures.as_completed(futures):
                if f.result():
                    successes += 1

        log.info(f"Created {successes}/{num_frames} concatenated frames in {temp_dir}")
        
        # Calculate fps to match RGB video duration
        rgbd_fps = fps
        if rgb_props and rgb_props['duration'] > 0:
            rgbd_fps = num_frames / rgb_props['duration']
            log.info(f"Using fps={rgbd_fps:.2f} to match RGB video duration ({num_frames} frames / {rgb_props['duration']:.2f}s)")
        
        # Create video from concatenated frames
        output_rgbd_video = output_dir / "rgbd.mp4"
        success = create_video_from_frames(
            temp_dir,
            output_rgbd_video,
            fps=rgbd_fps,
            pattern="*.png",
            is_depth=False,  # Frames are already processed
            web_compatible=True
        )
        
        if success:
            log.info(f"Successfully created side-by-side RGBD video: {output_rgbd_video}")
            # Verify RGBD video properties
            rgbd_props = get_video_properties(output_rgbd_video)
            if rgbd_props:
                log.info(f"RGBD video: duration={rgbd_props['duration']:.2f}s, fps={rgbd_props['fps']:.2f}, frames={rgbd_props['frame_count']}")
        
        return success
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            log.info(f"Cleaned up temporary directory: {temp_dir}")

