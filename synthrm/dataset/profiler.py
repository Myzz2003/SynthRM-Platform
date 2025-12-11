from argparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
from glob import glob
from PIL import Image
import imageio as iio
import numpy as np
import json
import io
import os

@dataclass
class _dir_entry:
    root: str
    block: int
    cam_dir: str = "camera"
    rgb_dir: str = "rgb"
    depth_dir: str = "depth"
    rm_dir: str = "config1/radiomap/rendered"
    tx_dir: str = "config1/txpos"

@dataclass
class _meta_path_entry:
    frame_paths: List[str]
    cam_paths: List[str]
    depth_paths: List[str]
    rgb_dir: str
    cam_dir: str
    depth_dir: str

@dataclass
class _rm_path_entry:
    rm_paths: List[str]
    tx_paths: List[str]
    rm_dir: str
    tx_dir: str

def loggit(message: str, **kwargs) -> None:
    print(f"\033[36m[profiler]\033[0m {message}", **kwargs)

def parse_arguments() -> Namespace:
    p = ArgumentParser(description="Making a profile for SynthRM dataset")
    arguments = [
        ("--block", "-b", int, 4, "Block ID"),
        ("--output", "-o", str, "profile.json", "Output profile file"),
        ("--root", "-r", str, None, "Root directory of SynthRM dataset"),
        ("--cam_dir", '-C', str, "camera", "Camera directory name under each block"),
        ("--rgb_dir", '-R', str, "rgb", "Frame directory name under each block"),
        ("--depth_dir", '-D', str, "depth", "Depth directory name under each block"),
        ("--xmax", "-x", float, 20.0, "Maximum x distance between point1 and the x-nearest point2"),
        ("--ymax", "-y", float, 20.0, "Maximum y distance between point1 and the y-nearest point2"),
        ("--visualize", "-v", bool, False, "Visualize the distance heatmaps"),
        ("--show_idx", "-s", int, 70, "Index to show neighbors for visualization"),
    ]
    argnames: List[str] = []
    for arg in arguments:
        argname, shortname, atype, default, helpstr = arg
        if default is None:
            required = True
        else:
            required = False
        p.add_argument(
            argname,
            shortname,
            type=atype,
            default=default,
            required=required,
            help=helpstr,
        )
        argnames.append(argname.lstrip("-").replace("-", "_"))
    args = p.parse_args()
    return args, argnames

def compose_path(entry: _dir_entry) -> Tuple[_meta_path_entry, _rm_path_entry]:
    ws = os.path.join(entry.root, f"block_{entry.block:d}")
    rgb_dir = os.path.join(ws, entry.rgb_dir)
    cam_dir = os.path.join(ws, entry.cam_dir)
    depth_dir = os.path.join(ws, entry.depth_dir)
    rm_dir = os.path.join(ws, entry.rm_dir)
    tx_dir = os.path.join(ws, entry.tx_dir)
    cam_paths = sorted(glob(os.path.join(cam_dir, "*.npy")))
    frame_paths = sorted(glob(os.path.join(rgb_dir, "*.png")))
    depth_paths = sorted(glob(os.path.join(depth_dir, "*.exr")))
    rm_paths = sorted(glob(os.path.join(rm_dir, "*.npy")))
    tx_paths = sorted(glob(os.path.join(tx_dir, "*.npy")))
    if not cam_paths:
        raise FileNotFoundError(f"No camera files found in {cam_dir}")
    if not frame_paths:
        raise FileNotFoundError(f"No RGB files found in {rgb_dir}")
    if not depth_paths:
        raise FileNotFoundError(f"No depth files found in {depth_dir}")
    return _meta_path_entry(
        frame_paths=frame_paths, 
        cam_paths=cam_paths, 
        depth_paths=depth_paths, 
        rgb_dir=rgb_dir,
        cam_dir=cam_dir, 
        depth_dir=depth_dir,
        ), _rm_path_entry(rm_paths=rm_paths, tx_paths=tx_paths, rm_dir=rm_dir, tx_dir=tx_dir)

def compose_rm_dict(rm_entry: _rm_path_entry, prefix: str = "radio_map_rendered", tx_ind: str = "tx") -> Dict[str, List[str]]:
    rm_profile: Dict[str, List[str]] = {}
    rm_profile["prefix"] = prefix
    rm_profile["tx_ind"] = tx_ind
    subdict: Dict[str, List[str]] = {}
    for path in rm_entry.rm_paths:
        fname = os.path.basename(path)
        fileIds = fname.replace(f"{prefix}_", "").replace(f"{tx_ind}_", "").replace(".npy", "").split("_")
        if fileIds[0] not in subdict:
            subdict[fileIds[0]] = []
        subdict[fileIds[0]].append(fname)
    rm_profile["rmfiles"] = subdict
    return rm_profile

def load_cam(cam_paths: List[str]) -> np.ndarray:
    cam_pos: List[np.ndarray] = []
    for cam_path in cam_paths:
        cam = np.load(cam_path)
        pos = cam[:3, 3]
        cam_pos.append(pos)
    return np.array(cam_pos)

def show_cam_pos(cam_pos: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.scatter(cam_pos[:, 0], cam_pos[:, 1], c='blue', marker='o')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Camera Positions')
    plt.grid()

def cal_dim_dist(cam_pos: np.ndarray, dim: int) -> np.ndarray:
    sliced = cam_pos[:, dim][:, np.newaxis]
    dist = np.abs(sliced - sliced.T)
    return dist

def show_dist_heatmap(dist: np.ndarray, title: str=None) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(dist, cmap='hot', interpolation='nearest')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.xlabel('Camera Index')
    plt.ylabel('Camera Index')

def filter_dist(xdist: np.ndarray, ydist: np.ndarray, max_dist: Tuple[float, float]) -> np.ndarray:
    x_filtered = xdist <= max_dist[0]
    y_filtered = ydist <= max_dist[1]
    filtered = np.logical_and(x_filtered, y_filtered)
    return filtered

def show_result(frame_paths: List[str], neighbor_idx: np.ndarray | List[int], max_col: int = 6) -> None:
    images: List[np.ndarray] = []
    for idx in neighbor_idx:
        frame_path = frame_paths[idx]
        img = iio.v3.imread(frame_path)
        images.append(img)
    rows: List[np.ndarray] = []
    _row: List[np.ndarray] = []
    _need_padding = (len(images) % max_col != 0)
    for i, img in enumerate(images):
        i += 1
        _row.append(img)
        if i % max_col == 0 or i == len(images):
            row_concat = np.concatenate(_row, axis=1)
            rows.append(row_concat)
            _row = []
    if _need_padding:
        n_padding = max_col - (len(images) % max_col)
        h, w, c = images[0].shape
        padding_img = np.zeros((h, w, c), dtype=np.uint8)
        for _ in range(n_padding):
            _row.append(padding_img)
        row_concat = np.concatenate(_row, axis=1)
        rows.append(row_concat)
    concatenated = np.concatenate(rows, axis=0)
    img = Image.fromarray(concatenated)
    MAX_SHAPE = 2048
    if max(img.size) > MAX_SHAPE:
        scale = MAX_SHAPE / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size)
    img.show()

def result_to_dict(paths: List[str], neighbors: List[np.ndarray]) -> Dict[str, List[str]]:
    '''
    Docstring for result_to_dict
    
    :param paths: The list of file paths, could either be frame paths or camera paths
    :type paths: List[str]
    :param neighbors: The neighboring indices for each frame/camera
    :type neighbors: List[nps.ndarray]
    :return: A dictionary mapping each path to its list of neighboring paths
    :rtype: Dict[str, List[str]]
    '''
    result: Dict[str, List[str]] = {}
    for i, path in enumerate(paths):
        neighbor_indices = neighbors[i]
        neighbor_paths = [paths[idx] for idx in neighbor_indices]
        result[path] = neighbor_paths
    return result

def abs2rel_path(paths: List[str], dir_name: str) -> List[str]:
    return [path.replace(dir_name + os.sep, "") for path in paths]

def main() -> None:
    args, argnames = parse_arguments()
    for argname in argnames:
        loggit(f"{argname}: {getattr(args, argname)}")

    path_ent, rm_ent = compose_path(_dir_entry(root=args.root, block=args.block, cam_dir=args.cam_dir, rgb_dir=args.rgb_dir))
    frame_paths, cam_paths, depth_paths, rgb_dir, cam_dir, depth_dir = \
        path_ent.frame_paths, path_ent.cam_paths, path_ent.depth_paths, path_ent.rgb_dir, path_ent.cam_dir, path_ent.depth_dir
    rm_paths, tx_paths, rm_dir, tx_dir = \
        rm_ent.rm_paths, rm_ent.tx_paths, rm_ent.rm_dir, rm_ent.tx_dir
    rm_profile = compose_rm_dict(rm_ent)
    loggit(f"Frame paths loaded: {len(frame_paths)}")
    loggit(f"Radiomap paths loaded: {len(rm_paths)}")
    loggit(f"TX positions loaded: {len(tx_paths)}")
    cam_pos = load_cam(cam_paths)
    loggit(f"Camera positions loaded: {cam_pos.shape}")
    x_dist = cal_dim_dist(cam_pos, dim=0)
    y_dist = cal_dim_dist(cam_pos, dim=1)
    filtered = filter_dist(x_dist, y_dist, (args.xmax, args.ymax))
    neighbors: List[np.ndarray] = []
    for pdist in filtered:
        neighbor_indices = np.where(pdist)[0].tolist()
        neighbors.append(neighbor_indices)
    if args.visualize:
        show_cam_pos(cam_pos)
        show_dist_heatmap(filtered.astype(int), title="Filtered Distance Heatmap")
        selected_cam_pos = [cam_pos[i] for i in neighbors[0]]
        selected_cam_pos = np.array(selected_cam_pos)
        show_cam_pos(selected_cam_pos)
        show_result(frame_paths, neighbors[args.show_idx])
        plt.show()
    frame_paths = \
        abs2rel_path(frame_paths, rgb_dir)
    cam_paths = \
        abs2rel_path(cam_paths, cam_dir)
    depth_paths = \
        abs2rel_path(depth_paths, depth_dir)
    tx_paths = \
        abs2rel_path(tx_paths, tx_dir)
    frame_profile = result_to_dict(frame_paths, neighbors)
    camera_profile = result_to_dict(cam_paths, neighbors)
    depth_profile = result_to_dict(depth_paths, neighbors)
    tx_profile = result_to_dict(tx_paths, neighbors)
    profile_dict = {
        "frame_dir": os.path.abspath(rgb_dir),
        "camera_dir": os.path.abspath(cam_dir),
        "depth_dir": os.path.abspath(depth_dir),
        "rm_dir": os.path.abspath(rm_dir),
        "tx_dir": os.path.abspath(tx_dir),
        "frame": frame_profile,
        "camera": camera_profile,
        "depth": depth_profile,
        "radiomap": rm_profile,
        "tx_pos": tx_profile,
        "meta": {
            "xmax": args.xmax,
            "ymax": args.ymax,
        }
    }
    with open(args.output, "w", encoding='ascii') as f:
        json.dump(profile_dict, f, indent=4)
    loggit(f"Profile saved to {args.output}")
