import os
import sys
import time
import colorsys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.video import normalize_video_rotation
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False


def should_disable_global_slam(cfg):
    if not _run_global:
        return True

    if cfg.DEVICE.lower() != 'cuda' or not torch.cuda.is_available():
        return False

    if os.name != 'nt':
        return False

    device_props = torch.cuda.get_device_properties('cuda')
    if device_props.major >= 12:
        logger.warning(
            'Disable DPVO global SLAM on Windows for CUDA capability '
            f'{device_props.major}.{device_props.minor}. '
            'The bundled DPVO runtime is not stable on this configuration and can '
            'terminate the interpreter without a Python traceback. '
            'Falling back to local-only mode.'
        )
        return True

    return False

def prepare_cfg(
        fast=False,
        pose_backend=None,
        pose_model=None,
        detector_ckpt=None,
        detector_imgsz=None,
        detector_scale=None,
        feature_batch_size=None,
        min_track_frames=None,
    ):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')

    if fast:
        cfg.FLIP_EVAL = False
        cfg.PREPROC.DETECTOR_CKPT = 'yolov8n.pt'
        cfg.PREPROC.DETECTOR_IMGSZ = 512
        cfg.PREPROC.DETECTOR_SCALE = 0.5
        cfg.PREPROC.FEATURE_BATCH_SIZE = 64

    if pose_backend is not None:
        cfg.PREPROC.POSE_BACKEND = pose_backend
    if pose_model is not None:
        cfg.PREPROC.POSE_MODEL = pose_model
    if detector_ckpt is not None:
        cfg.PREPROC.DETECTOR_CKPT = detector_ckpt
    if detector_imgsz is not None:
        cfg.PREPROC.DETECTOR_IMGSZ = int(detector_imgsz)
    if detector_scale is not None:
        cfg.PREPROC.DETECTOR_SCALE = float(detector_scale)
    if feature_batch_size is not None:
        cfg.PREPROC.FEATURE_BATCH_SIZE = int(feature_batch_size)
    if min_track_frames is not None:
        cfg.PREPROC.MIN_TRACK_FRAMES = int(min_track_frames)

    return cfg

def load_video(video):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    return cap, fps, length, width, height


class WHAM_API(object):
    def __init__(
            self,
            fast=False,
            pose_backend=None,
            pose_model=None,
            detector_ckpt=None,
            detector_imgsz=None,
            detector_scale=None,
            feature_batch_size=None,
            min_track_frames=None,
        ):
        self.cfg = prepare_cfg(
            fast=fast,
            pose_backend=pose_backend,
            pose_model=pose_model,
            detector_ckpt=detector_ckpt,
            detector_imgsz=detector_imgsz,
            detector_scale=detector_scale,
            feature_batch_size=feature_batch_size,
            min_track_frames=min_track_frames,
        )
        if self.cfg.DEVICE.lower() == 'cuda' and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.network = build_network(self.cfg, build_body_model(self.cfg.DEVICE, self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN))
        self.network.eval()
        self.detector = DetectionModel(
            self.cfg.DEVICE.lower(),
            bbox_model_ckpt=self.cfg.PREPROC.DETECTOR_CKPT,
            bbox_imgsz=self.cfg.PREPROC.DETECTOR_IMGSZ,
            detect_scale=self.cfg.PREPROC.DETECTOR_SCALE,
            pose_backend=self.cfg.PREPROC.POSE_BACKEND,
            pose_model_ckpt=self.cfg.PREPROC.POSE_MODEL,
            min_track_frames=self.cfg.PREPROC.MIN_TRACK_FRAMES,
        )
        self.extractor = FeatureExtractor(
            self.cfg.DEVICE.lower(),
            self.cfg.FLIP_EVAL,
            max_batch_size=self.cfg.PREPROC.FEATURE_BATCH_SIZE,
        )
        self.slam = None
    
    @torch.no_grad()
    def preprocessing(self, video, cap, fps, length, output_dir):
        if not (osp.exists(osp.join(output_dir, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_dir, 'slam_results.pth'))):
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                self.detector.track(img, fps, length)
                
                # SLAM
                if self.slam is not None: 
                    self.slam.track()

            tracking_results = self.detector.process(fps)
            
            if self.slam is not None: 
                slam_results = self.slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = self.extractor.run(video, tracking_results)
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_dir, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_dir, 'slam_results.pth'))
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_dir, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_dir, 'slam_results.pth'))

        return tracking_results, slam_results
    
    @torch.no_grad()
    def wham_inference(self, tracking_results, slam_results, width, height, fps, output_dir):
        # Build dataset
        dataset = CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps)
        
        # run WHAM
        results = defaultdict(dict)
        for batch in dataset:
            if batch is None: break

            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            
            # inference
            pred = self.network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
            
            # Store results
            results[_id]['poses_body'] = pred['poses_body'].cpu().squeeze(0).numpy()
            results[_id]['poses_root_cam'] = pred['poses_root_cam'].cpu().squeeze(0).numpy()
            results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
            results[_id]['verts_cam'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
            results[_id]['poses_root_world'] = pred['poses_root_world'].cpu().squeeze(0).numpy()
            results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
            results[_id]['frame_id'] = frame_id
        
        joblib.dump(slam_results, osp.join(output_dir, 'wham_results.pth'))
        return results
    
    @torch.no_grad()
    def __call__(self, video, output_dir='output/demo', calib=None, run_global=True, visualize=False):
        os.makedirs(output_dir, exist_ok=True)
        original_video = video
        video = normalize_video_rotation(video, output_dir)
        if video != original_video:
            for cached in ('tracking_results.pth', 'slam_results.pth'):
                p = osp.join(output_dir, cached)
                if osp.exists(p):
                    os.remove(p)
                    logger.info(f'Removed stale cache: {cached}')

        # load video information
        cap, fps, length, width, height = load_video(video)

        # Whether or not estimating motion in global coordinates
        run_global = run_global and not should_disable_global_slam(self.cfg)
        if run_global: self.slam = SLAMModel(video, output_dir, width, height, calib)
        
        # preprocessing to get detection, tracking, slam results and image features from video input
        tracking_results, slam_results = self.preprocessing(video, cap, fps, length, output_dir)

        # WHAM forward inference to get the results
        results = self.wham_inference(tracking_results, slam_results, width, height, fps, output_dir)
        
        # Visualize
        if visualize:
            from lib.vis.run_vis import run_vis_on_demo
            run_vis_on_demo(self.cfg, video, results, output_dir, self.network.smpl, vis_global=run_global)
        
        return results, tracking_results, slam_results


if __name__ == '__main__':
    #from wham_api import WHAM_API
    wham_model = WHAM_API()
    input_video_path = 'examples/IMG_9732.mov'
    results, tracking_results, slam_results = wham_model(input_video_path)
