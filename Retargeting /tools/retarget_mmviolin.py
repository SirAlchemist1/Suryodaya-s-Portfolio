#!/usr/bin/env python3
"""
Violin pose retargeting CLI tool.

This script retargets 3D pose data from source to target joint configurations,
supporting both single .npy trials and batched .npz datasets.

Usage:
    python tools/retarget_mmviolin.py \\
        --input <path to .npy or .npz> \\
        --mapping tools/mapping_example_38.json \\
        --outdir out/<name> \\
        [--save_individual]
"""

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Common human-skeleton joint counts we might see
KNOWN_JOINT_COUNTS: set[int] = {
    16, 17, 18, 19, 20, 21, 24, 25, 26, 28,
    30, 31, 32, 33, 34, 35, 36, 38, 49, 51, 55, 68
}

# Constants for NPZ parsing
POSE_KEYS = ["kp_optim_3d", "pose3d", "poses3d", "xyz", "joints3d", "kps3d", "kpts3d"]
PARTICIPANT_KEYS = ["P", "pid", "participant"]
CAMERA_KEYS = ["C", "camera", "cam", "view"]
TRIAL_KEYS = ["T", "trial", "take", "session", "kp_index_vid"]

# PCT regex pattern
PCT_PATTERN = r"(?:^|_)P(\d+)[-_]?C(\w+)[-_]?T(\d+)(?:_|$)"

def _move_coord_axis_last(a: np.ndarray) -> np.ndarray:
    """Ensure the coordinate axis (size == 3) is last; keep other axes' order."""
    if 3 not in a.shape:
        raise ValueError(f"Cannot find coordinate axis (size 3) in shape {a.shape}")
    if a.shape[-1] == 3:
        return a
    axis_3 = [i for i, s in enumerate(a.shape) if s == 3][0]
    perm = [i for i in range(a.ndim) if i != axis_3] + [axis_3]
    return np.transpose(a, perm)

def coerce_xyz(arr: np.ndarray, expected_joints: Optional[int] = None) -> np.ndarray:
    """
    Return array as (T, J, 3). Robustly handles common permutations.

    Heuristics:
      1) If array has more than 3 dimensions, squeeze or select first index.
      2) Push coordinate axis to last.
      3) If one non-coord axis equals `expected_joints`, that's J.
      4) Else if exactly one axis is in KNOWN_JOINT_COUNTS, that's J.
      5) Else if one axis is much larger (>64), that's T and the smaller is J.
      6) Else fallback: smaller axis is J, larger axis is T (then transpose to (T,J,3)).
    """
    a = np.asarray(arr)
    
    # Handle extra dimensions
    if a.ndim > 3:
        # If first dimension is 1, squeeze it
        if a.shape[0] == 1:
            a = np.squeeze(a, axis=0)
        else:
            # Take the first index
            a = a[0]
    
    if a.ndim != 3:
        raise ValueError(f"Expected a 3D array after squeezing/indexing, got shape {a.shape}")
        
    a = _move_coord_axis_last(a)  # -> (A, B, 3)
    A, B, _ = a.shape

    THRESH = 64  # time tends to be larger than this; joints usually smaller

    # 1) If expected joints is known, honor it.
    if expected_joints is not None:
        if A == expected_joints:
            # (J, T, 3) -> swap to (T, J, 3)
            return np.transpose(a, (1, 0, 2))
        if B == expected_joints:
            # Already (T, J, 3)
            return a

    # 2) If exactly one axis is a known joint count, use it.
    a_is_j = A in KNOWN_JOINT_COUNTS
    b_is_j = B in KNOWN_JOINT_COUNTS
    if a_is_j ^ b_is_j:  # XOR: exactly one is a joint count
        if a_is_j:
            # A is J -> swap to (T, J, 3)
            return np.transpose(a, (1, 0, 2))
        else:
            # B is J -> already (T, J, 3)
            return a

    # 3) If one axis is clearly larger (likely T), the smaller is J.
    if A > THRESH and B <= THRESH:
        # A is T, B is J -> already (T, J, 3)
        return a
    if B > THRESH and A <= THRESH:
        # B is T, A is J -> swap to (T, J, 3)
        return np.transpose(a, (1, 0, 2))

    # 4) Fallback: smaller dimension is J, larger is T.
    if A <= B:
        # A is J -> swap to (T, J, 3)
        return np.transpose(a, (1, 0, 2))
    else:
        # B is J -> already (T, J, 3)
        return a

def create_identity_mapping(J_src: int) -> Tuple[List[str], List[str], Dict, bool]:
    """
    Create an identity mapping for a given number of joints.
    
    Args:
        J_src: Number of source joints
        
    Returns:
        Tuple containing:
            - source_names: Generated source joint names
            - target_names: Same as source names
            - expanded_map: One-to-one index mapping
            - used_identity: Always True for identity mapping
    """
    source_names = [str(i) for i in range(J_src)]
    target_names = source_names.copy()
    expanded_map = {str(i): [i] for i in range(J_src)}
    return source_names, target_names, expanded_map, True

def load_mapping(json_path: str, J_src_hint: Optional[int] = None) -> Tuple[List[str], List[str], Dict, bool]:
    """
    Load and validate joint mapping configuration.
    
    Args:
        json_path: Path to mapping JSON file
        J_src_hint: Optional number of source joints to validate against
        
    Returns:
        Tuple containing:
            - source_names: List of source joint names
            - target_names: List of target joint names
            - expanded_map: Dict mapping target joints to source joint indices
            - used_identity: Whether identity mapping was used as fallback
    """
    with open(json_path) as f:
        config = json.load(f)
        
    source_names = config["source_names"]
    target_names = config["target_names"]
    mapping = config["map"]
    
    # Check joint count match
    if J_src_hint is not None and len(source_names) != J_src_hint:
        logger.warning(
            f"Mapping source joints ({len(source_names)}) doesn't match data joints ({J_src_hint}). "
            "Falling back to identity mapping."
        )
        return create_identity_mapping(J_src_hint)
    
    # Build index mapping
    src_name_to_idx = {name: i for i, name in enumerate(source_names)}
    expanded_map = {}
    
    for target, sources in mapping.items():
        if isinstance(sources, (int, str)):
            sources = [sources]
        try:
            if isinstance(sources[0], int):
                expanded_map[target] = sources
            else:
                expanded_map[target] = [src_name_to_idx[s] for s in sources]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid source joint reference in mapping: {e}")
            
    return source_names, target_names, expanded_map, False

def retarget_frames(src_xyz: np.ndarray, expanded_map: Dict, target_order: List[str]) -> np.ndarray:
    """
    Retarget source poses to target joint configuration.
    
    Args:
        src_xyz: Source poses with shape (T,J,3)
        expanded_map: Dict mapping target joints to source joint indices
        target_order: List of target joint names defining output order
        
    Returns:
        np.ndarray: Retargeted poses with shape (T,J_tgt,3)
    """
    T = src_xyz.shape[0]
    J_tgt = len(target_order)
    
    retargeted = np.full((T, J_tgt, 3), np.nan)
    warned_joints = set()  # Track which joints we've warned about
    
    for tgt_idx, target in enumerate(target_order):
        src_indices = expanded_map[target]
        src_points = src_xyz[:, src_indices, :]
        retargeted[:, tgt_idx, :] = np.nanmean(src_points, axis=1)
        
        # Check for all-nan frames
        nan_frames = np.all(np.isnan(retargeted[:, tgt_idx, :]), axis=1)
        if np.any(nan_frames) and target not in warned_joints:
            logger.warning(f"Joint '{target}' has {np.sum(nan_frames)} frames with all-nan values")
            warned_joints.add(target)
            
    return retargeted


# -------------------------
# Post-processing helpers
# -------------------------

def _find_pelvis_index(target_order: List[str]) -> Optional[int]:
    """Find a pelvis-like joint index in target_order (case-insensitive)."""
    if not target_order:
        return None
    names_lc = [n.lower() for n in target_order]
    for cand in ("pelvis", "root", "hip", "hips"):
        if cand in names_lc:
            return names_lc.index(cand)
    return None

def center_on_joint(xyz: np.ndarray, joint_idx: int) -> np.ndarray:
    """
    Subtract the joint coordinates from all joints per frame.
    xyz: (T,J,3)
    """
    if joint_idx is None or joint_idx < 0:
        return xyz
    ref = xyz[:, joint_idx:joint_idx+1, :]   # (T,1,3)
    return xyz - ref

def apply_unit(xyz: np.ndarray, unit: str) -> np.ndarray:
    """unit in {'mm','m'}; divide by 1000 for meters."""
    if unit == "m":
        return xyz / 1000.0
    return xyz

def _interp_nan_1d(y: np.ndarray, max_gap: int) -> None:
    """
    In-place linear interpolation on a 1D array for NaN gaps of length <= max_gap.
    Longer gaps remain NaN. Leading/trailing NaNs remain.
    """
    n = len(y)
    isn = np.isnan(y)
    if not isn.any():
        return
    # Indices
    xs = np.arange(n, dtype=float)
    # Known points
    known = ~isn
    if known.sum() == 0:
        return
    # Find contiguous NaN runs
    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        gap_len = j - i
        # interpolate only if gap is bounded on both sides and <= max_gap
        left = i - 1
        right = j
        if left >= 0 and right < n and gap_len <= max_gap and (not np.isnan(y[left])) and (not np.isnan(y[right])):
            # linear interp across [left, right]
            y[i:j] = np.interp(xs[i:j], [left, right], [y[left], y[right]])
        # else leave NaNs
        i = j

def interpolate_nans(xyz: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Apply per-joint, per-dimension linear interpolation for short NaN gaps.
    xyz: (T,J,3). Returns a new array.
    """
    out = xyz.copy()
    T, J, D = out.shape
    for j in range(J):
        for d in range(D):
            _interp_nan_1d(out[:, j, d], max_gap=max_gap)
    return out

def nan_fraction(xyz: np.ndarray) -> float:
    total = xyz.size
    nans = np.isnan(xyz).sum()
    return float(nans) / float(total) if total else 0.0

def bbox_nan_safe(xyz: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Return (min_x,min_y,min_z,max_x,max_y,max_z) ignoring NaNs.
    If all NaN along an axis, returns (nan, ...) for that axis.
    """
    mins = np.nanmin(xyz, axis=(0, 1))  # (3,)
    maxs = np.nanmax(xyz, axis=(0, 1))  # (3,)
    return (float(mins[0]), float(mins[1]), float(mins[2]),
            float(maxs[0]), float(maxs[1]), float(maxs[2]))

def _infer_pct_from_meta(meta: Optional[Dict]) -> Optional[str]:
    """
    Try to build PCT key from metadata dictionary.
    
    Args:
        meta: Dictionary containing participant/camera/trial metadata
        
    Returns:
        Optional[str]: PCT key if successful, None if required fields missing
    """
    if not meta:
        return None

    # Try various field names
    p = next((str(meta[k]) for k in PARTICIPANT_KEYS if k in meta), None)
    c = next((str(meta[k]) for k in CAMERA_KEYS if k in meta), None)
    t = next((str(meta[k]) for k in TRIAL_KEYS if k in meta), None)
    
    if all(x is not None for x in [p, c, t]):
        return f"P{int(p):02d}_C{c}_T{int(t):02d}"
    return None

def _infer_pct_from_key(key: str) -> Optional[str]:
    """
    Try to extract PCT components from a key string using regex.
    
    Args:
        key: String potentially containing P/C/T information
        
    Returns:
        Optional[str]: PCT key if pattern matched, None otherwise
    """
    match = re.search(PCT_PATTERN, key, re.IGNORECASE)
    if match:
        p, c, t = match.groups()
        return f"P{int(p):02d}_C{c}_T{int(t):02d}"
    return None

def as_pct_key(base_key: str, meta: Optional[Dict], fallback_idx: int) -> str:
    """
    Generate PCT key with fallbacks.
    
    Args:
        base_key: Original key from data source
        meta: Optional metadata dictionary
        fallback_idx: Index to use in worst-case fallback
        
    Returns:
        str: PCT key
    """
    if meta:
        pct = _infer_pct_from_meta(meta)
        if pct:
            return pct
            
    pct = _infer_pct_from_key(base_key)
    if pct:
        return pct
        
    logger.warning(f"Could not infer PCT for {base_key}, using fallback")
    return f"UNK_C_UNK_T_{fallback_idx:04d}"

def _move_coord_last_nd(a: np.ndarray) -> np.ndarray:
    """General version for ND arrays: move the axis with length 3 to the end."""
    if 3 not in a.shape:
        raise ValueError(f"Expected an axis of length 3 in shape {a.shape}")
    if a.shape[-1] == 3:
        return a
    axis_3 = [i for i, s in enumerate(a.shape) if s == 3][0]
    perm = [i for i in range(a.ndim) if i != axis_3] + [axis_3]
    return np.transpose(a, perm)

def _reorder_to_N_T_J_3(big: np.ndarray,
                        n_expected: int,
                        expected_joints: Optional[int]) -> np.ndarray:
    """
    big: coord-last 4D array (..., 3) to be permuted to (N, T, J, 3).
    Strategy:
      1) pick N-axis as the unique axis that equals n_expected (or closest).
      2) among remaining two, pick J-axis by expected_joints or known counts (else smaller).
      3) the leftover is T-axis.
    """
    if big.ndim != 4 or big.shape[-1] != 3:
        raise ValueError(f"Expected 4D coord-last array, got {big.shape}")
    d0, d1, d2, _ = big.shape
    dims = [d0, d1, d2]
    axes = [0, 1, 2]

    # 1) N-axis
    if dims.count(n_expected) == 1:
        n_axis = axes[dims.index(n_expected)]
    else:
        diffs = [abs(d - n_expected) for d in dims]
        n_axis = axes[diffs.index(min(diffs))]

    # remaining axes are (T, J)
    tj_axes = [ax for ax in axes if ax != n_axis]
    tj_sizes = [dims[ax] for ax in tj_axes]

    # 2) J-axis selection
    if expected_joints is not None and expected_joints in tj_sizes:
        j_axis = tj_axes[tj_sizes.index(expected_joints)]
    else:
        flags = [sz in KNOWN_JOINT_COUNTS for sz in tj_sizes]
        if sum(flags) == 1:
            j_axis = tj_axes[flags.index(True)]
        else:
            # smaller likely J
            j_axis = tj_axes[tj_sizes.index(min(tj_sizes))]

    # 3) T-axis = the other
    t_axis = [ax for ax in tj_axes if ax != j_axis][0]

    perm = [n_axis, t_axis, j_axis, 3]
    return np.transpose(big, perm)

def load_trials_npz(npz_path: str,
                    expected_joints: Optional[int] = None,
                    verbose_keys: bool = True) -> List[Tuple[str, np.ndarray, Optional[Dict]]]:
    """
    NPZ loader with three strategies:
      A) mmViolin format (kp_optim_3d + kp_index_vid)
      B) Big array + P/C/T lists
      C) Per-trial arrays under distinct keys (fallback)
    Returns: list of (pct_key, (T,J,3) ndarray, meta_or_None)
    """
    npz = np.load(npz_path, allow_pickle=True)
    files = list(npz.files)
    if verbose_keys:
        print(f"[load_trials_npz] keys: {files}")

    # Track PCT keys for deduplication
    seen_keys: Dict[str, Tuple[np.ndarray, Optional[Dict]]] = {}

    def add_trial(pct: str, xyz: np.ndarray, meta: Optional[Dict]) -> None:
        """Helper to add a trial with deduplication."""
        if pct in seen_keys:
            print(f"[dedupe] Duplicate key {pct}: keeping LAST (overwriting).")
        seen_keys[pct] = (xyz, meta)

    # Strategy A: mmViolin format
    if "kp_optim_3d" in files and "kp_optim_3d_vid" in files:
        print("[format] Detected mmViolin format")
        poses = npz["kp_optim_3d"]
        vid_indices = npz["kp_optim_3d_vid"]
        
        # Print initial data info
        print(f"[debug] poses type: {type(poses)}")
        print(f"[debug] vid_indices type: {type(vid_indices)}")
        
        # Handle scalar object arrays
        if isinstance(poses, np.ndarray) and poses.dtype == np.dtype('O'):
            print("[info] Converting poses object array")
            poses = poses.item()  # Get the actual object
            
        if isinstance(vid_indices, np.ndarray) and vid_indices.dtype == np.dtype('O'):
            print("[info] Converting video indices object array")
            vid_indices = vid_indices.item()
        
        # Handle dictionary objects
        if isinstance(poses, dict):
            print("[info] Converting poses dictionary")
            # Convert each frame to a numpy array first
            poses_list = []
            for frame in poses.values():
                if isinstance(frame, (list, np.ndarray)):
                    frame_arr = np.array(frame, dtype=np.float32)
                    poses_list.append(frame_arr)
            poses = poses_list
        elif isinstance(poses, (list, np.ndarray)):
            print("[info] Converting poses list/array")
            poses = [np.array(p, dtype=np.float32) if isinstance(p, (list, np.ndarray)) else p for p in poses]
            
        def to_scalar(x) -> int:
            """Convert any value to a scalar integer."""
            if isinstance(x, (list, np.ndarray)):
                # Take the first element if it's a sequence
                if len(x) > 0:
                    return to_scalar(x[0])
                return 0
            elif isinstance(x, np.integer):
                return int(x)
            elif isinstance(x, (int, float)):
                return int(x)
            else:
                try:
                    return int(x)
                except (TypeError, ValueError):
                    return 0

        if isinstance(vid_indices, dict):
            print("[info] Converting video indices dictionary")
            # Convert each index to a scalar
            vid_list = [to_scalar(idx) for idx in vid_indices.values()]
            vid_indices = np.array(vid_list, dtype=np.int32)
        elif isinstance(vid_indices, (list, np.ndarray)):
            print("[info] Converting video indices list/array")
            # Convert each index to a scalar
            vid_list = [to_scalar(idx) for idx in vid_indices]
            vid_indices = np.array(vid_list, dtype=np.int32)
        
        # Print converted data info
        print(f"[debug] Converted poses length: {len(poses)}")
        print(f"[debug] Converted vid_indices shape: {vid_indices.shape}")
        
        # Extract trial indices
        unique_indices = np.unique(vid_indices)
        print(f"[info] Found {len(unique_indices)} unique trials")
        
        # Group frames by video index
        trials_dict = {}
        for i, (frame, vid_idx) in enumerate(zip(poses, vid_indices)):
            if vid_idx not in trials_dict:
                trials_dict[vid_idx] = []
            trials_dict[vid_idx].append(frame)
        
        # Process each trial
        for i, (vid_idx, trial_xyz) in enumerate(trials_dict.items()):
            # Convert to float array if needed
            if len(trial_xyz) > 0:
                # Check if all frames have the same shape
                first_shape = trial_xyz[0].shape
                if all(frame.shape == first_shape for frame in trial_xyz):
                    trial_xyz = np.stack(trial_xyz)
                else:
                    print(f"[warning] Trial {i} has inconsistent frame shapes, skipping")
                    continue
            else:
                print(f"[warning] Trial {i} has no frames, skipping")
                continue
            
            # Print trial shape for debugging
            print(f"[debug] Trial {i} shape: {trial_xyz.shape}")
            
            # Ensure (T,J,3) format
            trial_xyz = coerce_xyz(trial_xyz, expected_joints=expected_joints)
            
            # Create PCT key
            pct = f"P01_C1_T{i+1:02d}"  # Default format if no other info available
            
            # Add metadata
            meta = {
                "participant": 1,
                "camera": 1,
                "trial": i+1,
                "video_index": int(vid_idx)
            }
            
            add_trial(pct, trial_xyz, meta)
        
        if seen_keys:
            return [(k, v[0], v[1]) for k, v in seen_keys.items()]

    # Strategy B: big array + P/C/T
    probable_big = [k for k in files
                    if isinstance(npz[k], np.ndarray)
                    and npz[k].ndim >= 3
                    and (3 in npz[k].shape)]
    P_key = next((k for k in files if k.lower() in ("p","pid","participant")), None)
    C_key = next((k for k in files if k.lower() in ("c","camera","cam","view")), None)
    T_key = next((k for k in files if k.lower() in ("t","trial","take","session")), None)

    if probable_big and P_key and C_key and T_key:
        print("[format] Detected big array + P/C/T format")
        P_list = np.array(npz[P_key]).ravel()
        C_list = np.array(npz[C_key]).ravel()
        T_list = np.array(npz[T_key]).ravel()
        n_expected = len(P_list)

        # Try to find a suitable array for poses
        big_key = None
        for k in probable_big:
            arr = npz[k]
            if arr.ndim == 4:  # (N,T,J,3) or similar
                big_key = k
                break
            elif arr.ndim == 3:  # (N,J,3) or similar
                if arr.shape[0] == n_expected:  # First dim matches trial count
                    big_key = k
                    break

        if big_key:
            big = npz[big_key]
            if big.ndim == 3:  # (N,J,3) or similar
                # For each trial, extract and coerce to (T,J,3)
                for i in range(n_expected):
                    trial_xyz = big[i]  # (J,3)
                    # Add a time dimension
                    trial_xyz = np.expand_dims(trial_xyz, axis=0)  # (1,J,3)
                    trial_xyz = coerce_xyz(trial_xyz, expected_joints=expected_joints)  # -> (T,J,3)
                    meta = {"participant": P_list[i], "camera": C_list[i], "trial": T_list[i]}
                    pct = _infer_pct_from_meta(meta)
                    if pct is None:
                        pct = f"UNK_C_UNK_T_{i:04d}"
                    add_trial(pct, trial_xyz, meta)
            else:  # 4D array
                big = _move_coord_last_nd(big)  # Move 3 to last axis
                big = _reorder_to_N_T_J_3(big, n_expected=n_expected, expected_joints=expected_joints)
                for i in range(n_expected):
                    meta = {"participant": P_list[i], "camera": C_list[i], "trial": T_list[i]}
                    pct = _infer_pct_from_meta(meta)
                    if pct is None:
                        pct = f"UNK_C_UNK_T_{i:04d}"
                    xyz = big[i]  # (T,J,3)
                    xyz = coerce_xyz(xyz, expected_joints=expected_joints)
                    add_trial(pct, xyz, meta)

    # Strategy C: per-trial arrays with keys
    if not seen_keys:
        print("[format] Trying per-trial arrays format")
        added = 0
        for k in files:
            if k.lower().startswith("meta"):
                continue
            arr = npz[k]
            if not isinstance(arr, np.ndarray) or arr.ndim != 3 or (3 not in arr.shape):
                continue
            xyz = coerce_xyz(arr, expected_joints=expected_joints)
            meta = None
            meta_k = f"meta_{k}"
            if meta_k in files:
                maybe = npz[meta_k].item()
                if isinstance(maybe, dict):
                    meta = maybe
            pct = _infer_pct_from_meta(meta) or _infer_pct_from_key(k) or f"UNK_C_UNK_T_{added:04d}"
            add_trial(pct, xyz, meta)
            added += 1

    if not seen_keys:
        raise RuntimeError("Could not parse trials from npz. Inspect keys / confirm format.")
    return [(k, v[0], v[1]) for k, v in seen_keys.items()]

def load_trials(path: str,
                expected_joints: Optional[int] = None) -> List[Tuple[str, np.ndarray, Optional[Dict]]]:
    """
    Unified loader for .npy and .npz.
    Returns: list of (pct_key, (T,J,3) ndarray, meta_or_None)
    """
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        xyz = coerce_xyz(arr, expected_joints=expected_joints)
        base = os.path.splitext(os.path.basename(path))[0]
        pct = _infer_pct_from_key(base) or "UNK_C_UNK_T_0000"
        return [(pct, xyz, None)]
    elif path.endswith(".npz"):
        return load_trials_npz(path, expected_joints=expected_joints, verbose_keys=True)
    else:
        raise ValueError("Input must be a .npy (single trial) or .npz (multi-trial).")

def save_outputs(outdir: str,
                 trials: List[Tuple[str, np.ndarray, Optional[Dict]]],
                 mapping: Tuple[List[str], List[str], Dict[str, List[int]], bool],
                 save_individual: bool,
                 dedupe: str = "last",
                 *,
                 center_pelvis: bool = False,
                 unit: str = "mm",
                 nan_policy: str = "keep",
                 interp_max_gap: int = 5,
                 min_frames: int = 1,
                 summary_csv: Optional[str] = None) -> None:
    """
    Retarget all trials using mapping and write NPZ + manifest; optionally per-trial NPY.
    mapping: (src_names, target_order, expanded_map, used_identity)
    """
    src_names, target_order, expanded_map, used_identity = mapping
    os.makedirs(outdir, exist_ok=True)

    store: Dict[str, np.ndarray] = {}
    seen: set[str] = set()
    dup_warned: set[str] = set()
    pel_idx = _find_pelvis_index(target_order) if center_pelvis else None

    # Prepare CSV writer if requested
    csv_fp = None
    csv_writer = None
    if summary_csv:
        os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
        csv_fp = open(summary_csv, "w", newline="")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "key", "T", "J_src_hint", "J_tgt",
            "used_identity", "nan_frac",
            "bbox_min_x", "bbox_min_y", "bbox_min_z",
            "bbox_max_x", "bbox_max_y", "bbox_max_z"
        ])

    for i, (pct, xyz, meta) in enumerate(trials):
        # retarget
        xyz_rt = retarget_frames(xyz, expanded_map, target_order)  # (T, J_tgt, 3)

        # Min-frames filter before further processing
        if xyz_rt.shape[0] < min_frames:
            print(f"[skip] {pct}: T={xyz_rt.shape[0]} < min_frames={min_frames}")
            continue

        # Post-processing: center → unit → NaN interpolation
        if pel_idx is not None:
            xyz_rt = center_on_joint(xyz_rt, pel_idx)
        xyz_rt = apply_unit(xyz_rt, unit)
        if nan_policy == "interp":
            xyz_rt = interpolate_nans(xyz_rt, interp_max_gap)

        # dedupe handling
        if pct in store:
            if dedupe == "first":
                if pct not in dup_warned:
                    print(f"[dedupe] Duplicate key {pct}: keeping FIRST, skipping later trial.")
                    dup_warned.add(pct)
                continue
            elif dedupe == "last":
                if pct not in dup_warned:
                    print(f"[dedupe] Duplicate key {pct}: keeping LAST (overwriting).")
                    dup_warned.add(pct)
                # fall through to overwrite
            elif dedupe == "error":
                raise RuntimeError(f"Duplicate PCT key detected: {pct}")

        store[pct] = xyz_rt
        if save_individual:
            np.save(os.path.join(outdir, f"{pct}.npy"), xyz_rt)

        # CSV summary row
        if csv_writer:
            nf = nan_fraction(xyz_rt)
            bminx, bminy, bminz, bmaxx, bmaxy, bmaxz = bbox_nan_safe(xyz_rt)
            csv_writer.writerow([
                pct, xyz_rt.shape[0], len(src_names) if src_names else "",
                len(target_order), used_identity, f"{nf:.6f}",
                f"{bminx:.6f}", f"{bminy:.6f}", f"{bminz:.6f}",
                f"{bmaxx:.6f}", f"{bmaxy:.6f}", f"{bmaxz:.6f}"
            ])

    # consolidated
    np.savez_compressed(os.path.join(outdir, "retargeted_trials.npz"), **store)

    manifest = {
        "target_names": target_order,
        "count": len(store),
        "keys": list(store.keys()),
        "used_identity_mapping": used_identity
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # summary
    example_keys = list(store.keys())[:5]
    print(f"[done] Saved {len(store)} trials → {outdir}")
    print(f"[done] Example keys: {example_keys}")
    print(f"[done] Mapping: J_src={'unknown' if src_names is None else len(src_names)} -> J_tgt={len(target_order)} "
          f"(identity={used_identity})")

    if csv_fp:
        csv_fp.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .npy (single trial) or .npz (multi-trial)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--mapping", required=True, help="JSON mapping file (source->target)")
    ap.add_argument("--save_individual", action="store_true", help="Also save per-trial .npy files")
    ap.add_argument("--dedupe", choices=("first","last","error"), default="last",
                    help="Policy for duplicate PCT keys (default: last)")
    # Post-processing flags
    ap.add_argument("--center_pelvis", action="store_true",
                    help="Center all frames on the pelvis/root joint if present in target_names")
    ap.add_argument("--unit", choices=("mm","m"), default="mm",
                    help="Coordinate unit conversion (mm=identity, m=divide by 1000)")
    ap.add_argument("--nan_policy", choices=("keep","interp"), default="keep",
                    help="Keep NaNs as-is or interpolate gaps up to --interp_max_gap")
    ap.add_argument("--interp_max_gap", type=int, default=5,
                    help="Max consecutive NaN frames to linearly interpolate per joint (default 5)")
    ap.add_argument("--min_frames", type=int, default=1,
                    help="Skip trials shorter than this length (default 1)")
    ap.add_argument("--summary_csv", type=str, default=None,
                    help="Optional CSV path for per-trial summary stats")
    args = ap.parse_args()

    # 1) Load trials first with flexible coercion (expected_joints unknown).
    trials = load_trials(args.input, expected_joints=None)
    if not trials:
        raise RuntimeError("No trials found in input.")

    # 2) Determine J_src_hint from the FIRST trial (post-coercion).
    J_src_hint = trials[0][1].shape[1]
    print(f"[hint] First trial shape: {trials[0][1].shape} -> J_src_hint={J_src_hint}")

    # 3) Load mapping (identity fallback if mismatch).
    src_names, target_order, expanded_map, used_identity = load_mapping(args.mapping, J_src_hint)

    # 4) Retarget & save.
    save_outputs(
        outdir=args.outdir,
        trials=trials,
        mapping=(src_names, target_order, expanded_map, used_identity),
        save_individual=args.save_individual,
        dedupe=args.dedupe,
        center_pelvis=args.center_pelvis,
        unit=args.unit,
        nan_policy=args.nan_policy,
        interp_max_gap=args.interp_max_gap,
        min_frames=args.min_frames,
        summary_csv=args.summary_csv
    )

if __name__ == "__main__":
    main()