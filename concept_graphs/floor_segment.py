from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


def _parse_signed_axes(ordering: str) -> tuple[list[tuple[int, int]], dict[str, int]]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    signed_axes: list[tuple[int, int]] = []
    axis_to_out_idx: dict[str, int] = {}

    sign = 1
    for ch in ordering:
        if ch == "-":
            sign = -1
            continue

        axis = ch.lower()
        if axis not in axis_map:
            raise ValueError(f"Invalid axis character '{ch}' in ordering '{ordering}'")

        out_idx = len(signed_axes)
        signed_axes.append((axis_map[axis], sign))
        axis_to_out_idx[axis] = out_idx
        sign = 1

    if len(signed_axes) != 3:
        raise ValueError("axes_ordering must specify exactly 3 axes")

    return signed_axes, axis_to_out_idx


def segment_floor_points(
    points: np.ndarray,
    *,
    axes_ordering: str = "xyz",
    floor_axis: str = "y",
    extension_tolerance_scale: float = 0.02,
    extension_tolerance_min: float = 0.05,
) -> np.ndarray:
    """Return a boolean mask for floor points from a point cloud.

    The floor is estimated as a dominant plane among points in the low-height
    region of the chosen vertical axis.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an array of shape (N, 3)")
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)

    floor_axis = floor_axis.lower()
    if floor_axis not in {"x", "y", "z"}:
        raise ValueError("floor_axis must be one of: 'x', 'y', 'z'")

    _ = _parse_signed_axes(axes_ordering)
    up_axis_idx = {"x": 0, "y": 1, "z": 2}[floor_axis]

    heights = points[:, up_axis_idx]
    low_quantile = float(np.quantile(heights, 0.20))
    low_mask = heights <= low_quantile

    if int(np.count_nonzero(low_mask)) < 64:
        low_mask = heights <= float(np.quantile(heights, 0.35))

    low_points = points[low_mask]
    if low_points.shape[0] < 16:
        return np.zeros((len(points),), dtype=bool)

    bounds = np.ptp(points, axis=0)
    scene_scale = float(np.linalg.norm(bounds))
    plane_distance = max(scene_scale * 0.005, 0.01)

    low_pcd = o3d.geometry.PointCloud()
    low_pcd.points = o3d.utility.Vector3dVector(low_points.astype(np.float64, copy=False))

    plane_model, inliers = low_pcd.segment_plane(
        distance_threshold=plane_distance,
        ransac_n=3,
        num_iterations=1000,
    )

    floor_mask = np.zeros((len(points),), dtype=bool)
    if len(inliers) < 16:
        log.warning("RANSAC found only %d inliers (need 16), no floor detected", len(inliers))
        return floor_mask

    # Always mark RANSAC inliers as floor as a baseline
    low_global_indices = np.flatnonzero(low_mask)
    low_inlier_indices = low_global_indices[np.asarray(inliers, dtype=np.int64)]
    floor_mask[low_inlier_indices] = True

    # plane_model is [a, b, c, d] defining ax + by + cz + d = 0
    a, b, c, d = [float(v) for v in plane_model]
    normal = np.array([a, b, c], dtype=np.float64)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-8:
        log.warning("Degenerate plane normal, returning RANSAC inliers only")
        return floor_mask

    normal /= norm_len
    d /= norm_len

    log.info(
        "Floor plane: normal=[%.4f, %.4f, %.4f], d=%.4f, "
        "scene_scale=%.3f, RANSAC inliers=%d",
        *normal, d, scene_scale, len(inliers),
    )

    # Perpendicular point-to-plane distance for all points.
    # Since normal is unit-length, this is distance along the plane normal.
    distances = np.abs(points.astype(np.float64) @ normal + d)

    tolerance = max(scene_scale * extension_tolerance_scale, extension_tolerance_min)
    plane_mask = distances <= tolerance

    q50, q75, q90, q95 = np.quantile(distances, [0.5, 0.75, 0.9, 0.95])

    log.info(
        "Floor extension: tolerance=%.4f (scale=%.4f,min=%.4f), "
        "points on plane=%d / %d, distance quantiles=[p50=%.4f,p75=%.4f,p90=%.4f,p95=%.4f]",
        tolerance, extension_tolerance_scale, extension_tolerance_min,
        int(plane_mask.sum()), len(points), q50, q75, q90, q95,
    )

    floor_mask |= plane_mask

    return floor_mask
