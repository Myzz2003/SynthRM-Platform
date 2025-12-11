import json
import os
import imageio as iio
import numpy as np
import sys
import warnings
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Any
from loguru import logger as loguruLog
from .profiler import _rm_path_entry as rmpath, _meta_path_entry as metapath
from .utils import create_object, timeit

def no_deprecation_warnings(func):
    '''
    Decorator to suppress warnings in the function.
    '''
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            loguruLog.debug(f"Suppressing warnings in {func.__name__}")
            return func(*args, **kwargs)
    return wrapper

@dataclass
class DataEntry:
    frame: np.ndarray
    camera: np.ndarray
    depth: np.ndarray
    txpos: np.ndarray = None
    rm: np.ndarray = None

    def __repr__(self) -> str:
        return f"""DataEntry(frame={self.frame.shape}, 
        camera={self.camera.shape}, depth={self.depth.shape}, 
        txpos={None if self.txpos is None else self.txpos.shape}, 
        rm={None if self.rm is None else self.rm.shape})"""

class BlockLoader:
    '''
    Docstring for BlockLoader
    '''
    def __init__(self, block: int, root: str, logger: Dict = None, profile: Dict = None, debug: bool = False): # type: ignore
        self._block = block
        self._root = root
        self._workspace = os.path.abspath(os.path.join(root, f"block_{block:d}"))
        if profile is None:
            profile = self._load_profile()
        else:
            profile = create_object(profile)
        self._profile = profile
        meta_path, rm_path = BlockLoader.parse_profile(profile)
        self._frame_paths, self._camera_paths, self._depth_paths, self._frame_dir, self._camera_dir, self._depth_dir = \
            meta_path.frame_paths, meta_path.cam_paths, meta_path.depth_paths, \
            meta_path.rgb_dir, meta_path.cam_dir, meta_path.depth_dir
        self._rm_files, self._tx_paths, self._rm_dir, self._tx_dir = \
            rm_path.rm_paths, rm_path.tx_paths, rm_path.rm_dir, rm_path.tx_dir
        if logger is None:
            self._logger = loguruLog.bind()
            self._logger.remove()
            self._logger.add(os.path.join(self._workspace, "debug_loader.log"), level="DEBUG", rotation="2 MB")
            self._logger.add(sys.stdout, level="INFO")
            self._logger.add(sys.stderr, level="ERROR")
        else:
            self._logger = create_object(logger)
        self.log("BlockLoader initialized.")
        self.log(len(self._frame_paths), "DEBUG")
        self.log(f"Length of block {self._count_len()}", "DEBUG")
        self._len = self._count_len()
        
    def _load_profile(self) -> Dict:
        profile_path = os.path.join(self._workspace, "profile.json")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile file not found at {profile_path}")
        with open(profile_path, "r", encoding='ascii') as f:
            profile = json.load(f)
        return profile
    
    def _count_len(self) -> int:
        c = 0
        for k,v in self._rm_files.items():
            c += len(v)
        return c
    
    def __len__(self) -> int:
        return self._len
    
    @property
    def N(self) -> int:
        return len(self._frame_paths)
    
    @property
    def workspace(self) -> str:
        return self._workspace
    
    @staticmethod
    def parse_profile(profile: Dict) -> Tuple[metapath, rmpath]:
        frame_dir = profile["frame_dir"]
        camera_dir = profile["camera_dir"]
        depth_dir = profile["depth_dir"]
        frame_fname = profile["frame"].keys()
        camera_fname = profile["camera"].keys()
        depth_fname = profile["depth"].keys()
        frame_paths: List[str] = [os.path.join(frame_dir, fname) for fname in frame_fname]
        camera_paths: List[str] = [os.path.join(camera_dir, fname) for fname in camera_fname]
        depth_paths: List[str] = [os.path.join(depth_dir, fname) for fname in depth_fname]
        meta_path = metapath(
            frame_paths=frame_paths,
            cam_paths=camera_paths,
            depth_paths=depth_paths,
            rgb_dir=frame_dir,
            cam_dir=camera_dir,
            depth_dir=depth_dir
        )
        rm_dir = profile["rm_dir"]
        tx_dir = profile["tx_dir"]
        # This contains a dict mapping file index to all tx positions (10 positions each) files
        # (e.g. '0000' -> ['radio_map_rendered_0000_tx_01.npy', 'radio_map_rendered_0000_tx_02.npy', ...])
        rm_files = profile["radiomap"]["rmfiles"]
        tx_files = profile["tx_pos"]
        rm_path = rmpath(
            rm_paths=rm_files,
            tx_paths=[os.path.join(rm_dir, fname) for fname in tx_files.keys()],
            rm_dir=rm_dir,
            tx_dir=tx_dir
        )
        return meta_path, rm_path
    
    @staticmethod
    def get_neighbors(profile: Dict, key: Tuple[str, str, str, str]) -> Tuple[List[str], List[str], List[str], List[str], List[int]]:
        frame_neighbors = profile["frame"][key[0]]
        camera_neighbors = profile["camera"][key[1]]
        depth_neighbors = profile["depth"][key[2]]
        tx_neighbors = profile["tx_pos"][key[3]]
        neighbor_idx = [int(frame_neighbors[i].split(".")[0]) for i in range(len(frame_neighbors))]
        return frame_neighbors, camera_neighbors, depth_neighbors, tx_neighbors, neighbor_idx
    
    @staticmethod
    def compose_path(ws: str, dir_name: str, fname: str | List[str]) -> str:
        if isinstance(fname, str):
            return os.path.join(ws, dir_name, fname)
        else:
            return [os.path.join(ws, dir_name, f) for f in fname]
        
    @staticmethod
    @timeit
    def load_image(paths: List[str], nsample: int = -1) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        if nsample > 0:
            paths = paths[:nsample]
        for path in paths:
            img = iio.v3.imread(path)
            images.append(img)
        return images
    
    @staticmethod
    @timeit
    def load_cam(paths: List[str], nsample: int = -1) -> np.ndarray:
        cams: List[np.ndarray] = []
        if nsample > 0:
            paths = paths[:nsample]
        for path in paths:
            cam = np.load(path)
            cams.append(cam)
        return np.array(cams)
    
    @staticmethod
    @no_deprecation_warnings
    @timeit
    def load_depth(paths: List[str], nsample: int = -1) -> List[np.ndarray]:
        depths: List[np.ndarray] = []
        if nsample > 0:
            paths = paths[:nsample]
        for path in paths:
            depth = iio.v3.imread(path)
            if depth.ndim == 2:
                depth = depth[:, :, np.newaxis]
            depths.append(depth)
        return depths
    
    @staticmethod
    @timeit
    def load_txpos(paths: List[str], tx_idx: int, nsample: int = -1) -> np.ndarray:
        if nsample > 0:
            paths = paths[:nsample]
        tx_positions: List[np.ndarray] = []
        for path in paths:
            tx_all = np.load(path)
            tx_positions.append(tx_all[tx_idx])
        return np.array(tx_positions)
    
    @staticmethod
    @timeit
    def load_rm(paths: List[str], nsample: int = -1) -> List[np.ndarray]:
        rms: List[np.ndarray] = []
        if nsample > 0:
            paths = paths[:nsample]
        for path in paths:
            rm = np.load(path)
            rms.append(rm)
        return rms
    
    @timeit
    def load_entry(self, idx: int, nsample: int = -1) -> DataEntry:
        frame_idx = idx // 10
        tx_idx = idx % 10
        frame = f"{frame_idx:04d}.png"
        camera = f"extrinsic_{frame_idx:04d}.npy"
        depth = f"{frame_idx:04d}.exr"
        tx = f"tx_pos_{frame_idx:04d}.npy"

        frame_neighbors, camera_neighbors, depth_neighbors, tx_neighbors, neighbor_idx = \
            BlockLoader.get_neighbors(self._profile, (frame, camera, depth, tx))
        
        rmIdx = [f"{i:04d}" for i in neighbor_idx]
        rmfile = [self._rm_files[idx][tx_idx] for idx in rmIdx]
        
        frame_paths = BlockLoader.compose_path(self._workspace, self._frame_dir, frame_neighbors)
        camera_paths = BlockLoader.compose_path(self._workspace, self._camera_dir, camera_neighbors)
        depth_paths = BlockLoader.compose_path(self._workspace, self._depth_dir, depth_neighbors)
        tx_paths = BlockLoader.compose_path(self._workspace, self._tx_dir, tx_neighbors)
        rm_paths = BlockLoader.compose_path(self._workspace, self._rm_dir, rmfile)
        frame_images = BlockLoader.load_image(frame_paths, nsample)
        camera_matrices = BlockLoader.load_cam(camera_paths, nsample)
        depth_images = BlockLoader.load_depth(depth_paths, nsample)
        tx_positions = BlockLoader.load_txpos(tx_paths, tx_idx, nsample)
        rm_images = BlockLoader.load_rm(rm_paths, nsample)

        return DataEntry(
            frame=np.array(frame_images),
            camera=camera_matrices,
            depth=np.array(depth_images),
            txpos=tx_positions,
            rm=np.array(rm_images)
        )

    def log(self, message: str, level: str = "INFO") -> None:
        self._logger.log(level, message)

if __name__ == "__main__":
    loader = BlockLoader(block=4, root="./")
    entry = loader.load_entry(0, 3)
    loader.log(f"Loaded entry: {entry}", "DEBUG")