#!/usr/bin/env python3
"""
Simple 3D viewer for pose data visualization.

Usage:
    python tools/viewer3d.py --npy path/to/trial.npy [--connectivity path/to/edges.json]
    python tools/viewer3d.py --npz path/to/trials.npz --key <PCT_key> [--connectivity path/to/edges.json]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def _set_equal_aspect(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5 * max_range
    Yb = 0.5 * max_range
    Zb = 0.5 * max_range
    ax.set_xlim(float(X.mean()-Xb), float(X.mean()+Xb))
    ax.set_ylim(float(Y.mean()-Yb), float(Y.mean()+Yb))
    ax.set_zlim(float(Z.mean()-Zb), float(Z.mean()+Zb))

def load_edges(path: Optional[str]) -> List[Tuple[int, int]]:
    """Load edge list from JSON file or return empty list."""
    if not path:
        return []
    with open(path) as f:
        return json.load(f)

def load_poses(npy_path: Optional[str] = None, npz_path: Optional[str] = None, key: Optional[str] = None) -> np.ndarray:
    """
    Load poses from either .npy file or specific key in .npz file.
    
    Args:
        npy_path: Path to .npy file
        npz_path: Path to .npz file
        key: Key to load from .npz file
        
    Returns:
        np.ndarray: Pose data with shape (T,J,3)
    """
    if npy_path:
        xyz = np.load(npy_path)
    elif npz_path and key:
        data = np.load(npz_path)
        if key not in data:
            raise ValueError(f"Key '{key}' not found in {npz_path}")
        xyz = data[key]
    else:
        raise ValueError("Must provide either --npy or both --npz and --key")
        
    if len(xyz.shape) != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Expected shape (T,J,3), got {xyz.shape}")
        
    return xyz

def plot_pose_frame(ax: Axes3D, xyz: np.ndarray, edges: List[Tuple[int, int]], frame_idx: int):
    """Plot a single pose frame as 3D scatter with optional edges."""
    # Plot joints
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='b', marker='o')
    
    # Plot edges if provided
    for j1, j2 in edges:
        if j1 >= xyz.shape[0] or j2 >= xyz.shape[0]:
            continue  # Skip invalid edge indices
        pts = xyz[[j1, j2]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-')
        
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Frame {frame_idx}")
    
    # Set equal aspect ratio and reasonable view
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)

def main():
    parser = argparse.ArgumentParser(description="3D pose viewer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npy", help="Path to .npy file with shape (T,J,3)")
    group.add_argument("--npz", help="Path to .npz file containing pose data")
    parser.add_argument("--key", help="Key to load from .npz file")
    parser.add_argument("--connectivity", help="Optional path to edge list JSON")
    parser.add_argument("--animate", type=int, nargs="?", const=-1, default=None,
                        help="Animate first N frames (or all if omitted)")
    parser.add_argument("--save_mp4", type=str, default=None, help="Save animation to MP4 path (requires ffmpeg)")
    parser.add_argument("--ground", action="store_true", help="Draw z=0 ground plane")
    args = parser.parse_args()
    
    # Validate args
    if args.npz and not args.key:
        raise ValueError("Must provide --key when using --npz")
        
    # Load data and edges
    xyz = load_poses(args.npy, args.npz, args.key)
    edges = load_edges(args.connectivity)
    
    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    T, J, _ = xyz.shape
    pts = ax.scatter(xyz[0,:,0], xyz[0,:,1], xyz[0,:,2])
    lines = []
    if edges:
        for (a,b) in edges:
            la, = ax.plot([xyz[0,a,0], xyz[0,b,0]],
                          [xyz[0,a,1], xyz[0,b,1]],
                          [xyz[0,a,2], xyz[0,b,2]])
            lines.append(la)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Equal aspect using first-frame bbox (robust enough for quick QC)
    _set_equal_aspect(ax, xyz[0,:,0], xyz[0,:,1], xyz[0,:,2])

    # Optional ground plane
    if args.ground:
        gmin, gmax = ax.get_xlim()[0], ax.get_xlim()[1]
        Xg, Yg = np.meshgrid(np.linspace(gmin, gmax, 2), np.linspace(gmin, gmax, 2))
        Zg = np.zeros_like(Xg)
        ax.plot_surface(Xg, Yg, Zg, alpha=0.1)

    if args.animate is None:
        plt.show()
        return

    # Animation mode
    N = T if args.animate == -1 else max(1, min(T, args.animate))

    def update(f):
        pts._offsets3d = (xyz[f,:,0], xyz[f,:,1], xyz[f,:,2])
        for li, (a,b) in enumerate(edges):
            lines[li].set_data([xyz[f,a,0], xyz[f,b,0]],
                               [xyz[f,a,1], xyz[f,b,1]])
            lines[li].set_3d_properties([xyz[f,a,2], xyz[f,b,2]])
        return [pts, *lines]

    ani = animation.FuncAnimation(fig, update, frames=N, interval=33, blit=False)
    if args.save_mp4:
        try:
            writer = animation.FFMpegWriter(fps=30, bitrate=1800)
            ani.save(args.save_mp4, writer=writer)
            print(f"[viewer] Saved animation → {args.save_mp4}")
        except Exception as e:
            print(f"[viewer] Could not save MP4 (ffmpeg missing?): {e}")
            plt.show()
    else:
        plt.show()

if __name__ == "__main__":
    main()