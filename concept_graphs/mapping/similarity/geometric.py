from typing import Tuple, List
import torch
import torch.nn.functional as F
from .Similarity import GeometricSimilarity


def point_cloud_overlap(
    pcd_1: torch.Tensor, pcd_2: torch.Tensor, eps: float
) -> Tuple[torch.Tensor]:
    # Create masks for valid points
    valid_1 = ~torch.isnan(pcd_1).any(dim=-1)  # (n1,)
    valid_2 = ~torch.isnan(pcd_2).any(dim=-1)  # (n2,)
    
    is_close = torch.cdist(pcd_1, pcd_2) < eps # (n1, n2)
    
    overlap_1 = is_close.any(dim=1).to(pcd_1.dtype)  # (n1,)
    overlap_2 = is_close.any(dim=0).to(pcd_2.dtype)  # (n2,)
    
    overlap_1 = torch.where(valid_1, overlap_1, torch.nan)
    overlap_2 = torch.where(valid_2, overlap_2, torch.nan)
    
    d1 = torch.nanmean(overlap_1)
    d2 = torch.nanmean(overlap_2)
    
    return d1, d2

batched_point_cloud_overlap = torch.vmap(point_cloud_overlap, in_dims=(0, None, None))

class PointCloudOverlapClosestK(GeometricSimilarity):
    """Point Cloud Overlap with closest k other point clouds in terms of centroid distance."""

    def __init__(self, eps: float, agg: str, k: int = 3, max_dist_centroid=10.0):
        self.eps = eps
        self.agg = agg
        self.k = max(2, k)
        self.max_dist_centroid = max_dist_centroid

    def __call__(
            self,
            main_pcd: List[torch.Tensor],
            main_centroid: torch.Tensor,
            other_pcd: List[torch.Tensor],
            other_centroid: torch.Tensor,
            is_symmetrical: bool,
        ) -> torch.Tensor:
            dist_centroids = torch.cdist(main_centroid, other_centroid)

            k = min(len(main_pcd), self.k)
            closest_k = torch.topk(dist_centroids, k=k, dim=0, largest=False).indices.T
            result = torch.zeros_like(dist_centroids)
            # Pad point cloud 
            max_n1 = max([pcd.shape[0] for pcd in main_pcd])
            padded_main_pcd = torch.stack([
                F.pad(seq, (0, 0, 0, max_n1 - seq.shape[0]), value=torch.nan)
                for seq in main_pcd
            ])

            # Iterate over each point cloud in other_pcd
            for other_i, other_pcd_i in enumerate(other_pcd):
                # Get k nearest neighbors for this other point cloud
                k_nearest_indices = closest_k[other_i]  # (k,)
                
                # Filter by distance
                dist_values = dist_centroids[k_nearest_indices, other_i]
                valid_mask = dist_values <= self.max_dist_centroid
                
                if not valid_mask.any():
                    continue  # Skip if no valid pairs
                
                valid_indices = k_nearest_indices[valid_mask]
                
                # Compute similarities using vmap
                sim1, sim2 = batched_point_cloud_overlap(
                    padded_main_pcd[valid_indices], 
                    other_pcd_i, 
                    self.eps
                )
                
                # Aggregate similarities
                if self.agg == "sum":
                    sim = sim1 + sim2
                elif self.agg == "mean":
                    sim = (sim1 + sim2) / 2
                elif self.agg == "max":
                    sim = torch.max(sim1, sim2)
                elif self.agg == "other":
                    sim = sim2
                else:
                    raise ValueError(f"Unknown aggregation method {self.agg}")
                
                # Handle symmetrical case for valid indices
                if is_symmetrical:
                    sym_mask = (valid_indices == other_i)
                    sim = torch.where(sym_mask, 1.0, sim)
                
                # Store results
                result[valid_indices, other_i] = sim
            
            return result


    # Fully batched version
    # def __call__(
    #     self,
    #     main_pcd: List[torch.Tensor],
    #     main_centroid: torch.Tensor,
    #     other_pcd: List[torch.Tensor],
    #     other_centroid: torch.Tensor,
    #     is_symmetrical: bool,
    # ) -> torch.Tensor:
    #     dist_centroids = torch.cdist(main_centroid, other_centroid)

    #     k = min(len(main_pcd), self.k)
    #     closest_k = torch.topk(dist_centroids, k=k, dim=0, largest=False).indices.T
    #     result = torch.zeros_like(dist_centroids)
    #     # Pad point clouds if they have less than max_points_pcd points
    #     max_n1 = max([pcd.shape[0] for pcd in main_pcd])
    #     max_n2 = max([pcd.shape[0] for pcd in other_pcd])
    #     padded_main = torch.stack([
    #         F.pad(seq, (0, 0, 0, max_n1 - seq.shape[0]), value=torch.nan)
    #         for seq in main_pcd
    #     ]).to(torch.float16)
    #     padded_other = torch.stack([
    #         F.pad(seq, (0, 0, 0, max_n2 - seq.shape[0]), value=torch.nan)
    #         for seq in other_pcd
    #     ]).to(torch.float16)

    #     n_other = len(other_pcd)
    #     other_indices = torch.arange(n_other).unsqueeze(1).expand(-1, k).reshape(-1)  # (n_other * k,)
    #     main_indices = closest_k.reshape(-1)  # (n_other * k,)

    #     selected_main = padded_main[main_indices]  # (n_other * k, max_n1, 3)
    #     selected_other = padded_other[other_indices]  # (n_other * k, max_n2, 3)

    #     point_cloud_overlap_batched = torch.vmap(point_cloud_overlap, in_dims=(0, 0, None))
    #     sim1, sim2 = point_cloud_overlap_batched(selected_main, selected_other, self.eps)

    #     # Aggregate similarities
    #     if self.agg == "sum":
    #         sim = sim1 + sim2
    #     elif self.agg == "mean":
    #         sim = (sim1 + sim2) / 2
    #     elif self.agg == "max":
    #         sim = torch.max(sim1, sim2)
    #     elif self.agg == "other":
    #         sim = sim2
    #     else:
    #         raise ValueError(f"Unknown aggregation method {self.agg}")

    #     # Handle symmetrical case
    #     if is_symmetrical:
    #         sym_mask = (main_indices == other_indices)
    #         sim = torch.where(sym_mask, torch.tensor(1.0), sim)

    #     # Handle max distance filter
    #     dist_values = dist_centroids[main_indices, other_indices]  # (n_other * k,)
    #     dist_mask = dist_values <= self.max_dist_centroid
    #     sim = torch.where(dist_mask, sim, torch.tensor(0.0))
        
    #     # Scatter results back to result matrix
    #     result = torch.zeros_like(dist_centroids)
    #     result[main_indices, other_indices] = sim.to(dist_centroids.dtype)

    #     return result

  

class PointCloudOverlap(GeometricSimilarity):
    """Point Cloud Overlap."""

    def __init__(self, eps: float, agg: str):
        self.eps = eps
        self.agg = agg

    def __call__(
        self,
        main_pcd: List[torch.Tensor],
        main_centroid: torch.Tensor,
        other_pcd: List[torch.Tensor],
        other_centroid: torch.Tensor,
        is_symmetrical: bool,
    ) -> torch.Tensor:

        result = torch.zeros(len(main_pcd), len(other_pcd), device=main_centroid.device)
        # Pad point cloud 
        max_points_main = max([pcd.shape[0] for pcd in main_pcd])
        padded_main_pcd = torch.stack([
            F.pad(seq, (0, 0, 0, max_points_main - seq.shape[0]), value=torch.nan)
            for seq in main_pcd
        ])
        main_indices = torch.arange(len(main_pcd))
        for other_i, other_pcd_i in enumerate(other_pcd):

            # Compute similarities using vmap
            sim1, sim2 = batched_point_cloud_overlap(
                padded_main_pcd, 
                other_pcd_i, 
                self.eps
            ) 

            if self.agg == "sum":
                sim = sim1 + sim2
            elif self.agg == "mean":
                sim = (sim1 + sim2) / 2
            elif self.agg == "max":
                sim = torch.max(sim1, sim2)
            elif self.agg == "other":
                sim = sim2
            else:
                raise ValueError(f"Unknown aggregation method {self.agg}")

            # Handle symmetrical case for valid indices
            if is_symmetrical:
                sym_mask = (main_indices == other_i)
                sim = torch.where(sym_mask, 1.0, sim)

            # Store results
            result[main_indices, other_i] = sim

        # torch.set_printoptions(precision=2)
        # print(result)

        return result
