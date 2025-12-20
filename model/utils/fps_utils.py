# GaussianFormer/model/utils/fps_utils.py
import torch
import numpy as np

try:
    from pointops import farthest_point_sampling as pointops_fps
    POINTOPS_AVAILABLE = True
    print("✅ Successfully imported farthest_point_sampling from pointops")
except ImportError as e:
    print(f"❌ pointops import failed: {e}")
    print("🔄 Using custom FPS implementation!!!!!!")
    POINTOPS_AVAILABLE = False

def farthest_point_sampling(xyz, offset, new_offset):
    """
    Unified FPS function that works with both pointops and custom implementation
    
    Args:
        xyz: (N, 3) tensor of point coordinates
        offset: tensor of original point counts per batch
        new_offset: tensor of target point counts per batch
    
    Returns:
        indices: sampled point indices
    """
    if POINTOPS_AVAILABLE:
        return pointops_fps(xyz, offset, new_offset)
    else:
        return custom_farthest_point_sampling(xyz, offset, new_offset)

def custom_farthest_point_sampling(xyz, offset, new_offset):
    """
    Custom implementation of farthest point sampling
    
    Args:
        xyz: (N, 3) tensor of point coordinates
        offset: tensor of original point counts (for compatibility)
        new_offset: tensor of target point counts
    
    Returns:
        indices: (M) tensor of sampled point indices
    """
    # Handle different input types
    if isinstance(offset, torch.Tensor):
        if offset.numel() == 1:
            n_points = offset.item()
        else:
            # For batched processing, we assume single batch for simplicity
            n_points = xyz.shape[0]
    else:
        n_points = offset
        
    if isinstance(new_offset, torch.Tensor):
        if new_offset.numel() == 1:
            n_samples = new_offset.item()
        else:
            # For batched processing, take the last element as total samples
            n_samples = new_offset[-1].item() if new_offset.numel() > 1 else new_offset.item()
    else:
        n_samples = new_offset
    
    # If we need more samples than available points, return all indices
    if n_samples >= n_points:
        return torch.arange(n_points, device=xyz.device)
    
    # Initialize
    selected_indices = torch.zeros(n_samples, dtype=torch.long, device=xyz.device)
    distances = torch.ones(n_points, device=xyz.device) * 1e10
    
    # Start with a random point
    start_idx = torch.randint(0, n_points, (1,), device=xyz.device)
    selected_indices[0] = start_idx
    
    # Iteratively select the farthest point
    for i in range(1, n_samples):
        last_selected = selected_indices[i-1]
        # Compute distances to the last selected point
        new_dist = torch.sum((xyz - xyz[last_selected]) ** 2, dim=1)
        # Update minimum distances
        distances = torch.min(distances, new_dist)
        # Select the point with the maximum minimum distance
        selected_indices[i] = torch.argmax(distances)
    
    return selected_indices

def batched_farthest_point_sampling(xyz_list, n_samples_list):
    """
    Batched version of FPS for multiple point clouds
    
    Args:
        xyz_list: list of (N_i, 3) tensors
        n_samples_list: list of target sample counts
    
    Returns:
        indices_list: list of sampled indices for each point cloud
    """
    indices_list = []
    for xyz, n_samples in zip(xyz_list, n_samples_list):
        offset = torch.tensor([xyz.shape[0]], device=xyz.device, dtype=torch.int)
        new_offset = torch.tensor([n_samples], device=xyz.device, dtype=torch.int)
        indices = farthest_point_sampling(xyz, offset, new_offset)
        indices_list.append(indices)
    return indices_list

# Alternative implementations for different use cases
def fps_simple(xyz, n_samples):
    """
    Simplified FPS interface for single point cloud
    
    Args:
        xyz: (N, 3) point cloud
        n_samples: number of points to sample
    
    Returns:
        indices: (n_samples) sampled indices
    """
    return farthest_point_sampling(xyz, xyz.shape[0], n_samples)

def fps_random_fallback(xyz, n_samples):
    """
    Fallback to random sampling when FPS is too slow
    
    Args:
        xyz: (N, 3) point cloud
        n_samples: number of points to sample
    
    Returns:
        indices: (n_samples) randomly sampled indices
    """
    n_points = xyz.shape[0]
    if n_samples >= n_points:
        return torch.arange(n_points, device=xyz.device)
    else:
        return torch.randperm(n_points, device=xyz.device)[:n_samples]