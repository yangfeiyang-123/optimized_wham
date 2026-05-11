from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import cv2
import torch
import numpy as np
import scipy.signal as signal
from progress.bar import Bar
from scipy.ndimage.filters import gaussian_filter1d

from configs import constants as _C
from .backbone.hmr2 import hmr2
from .backbone.utils import process_image
from ...utils.imutils import flip_kp, flip_bbox

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")

class FeatureExtractor(object):
    def __init__(self, device, flip_eval=False, max_batch_size=64):
        
        self.device = device
        self.flip_eval = flip_eval
        self.max_batch_size = max_batch_size
        
        ckpt = osp.join(ROOT_DIR, 'checkpoints', 'hmr2a.ckpt')
        self.model = hmr2(ckpt).to(device).eval()

    def _iter_frame_subjects(self, tracking_results, frame_id):
        subjects = []
        for _id, val in tracking_results.items():
            if frame_id not in val['frame_id']:
                continue
            frame_id2 = np.where(val['frame_id'] == frame_id)[0][0]
            subjects.append((_id, frame_id2, val))
        return subjects

    @torch.no_grad()
    def _encode_batch(self, batch):
        outputs = []
        for start in range(0, len(batch), self.max_batch_size):
            chunk = batch[start:start + self.max_batch_size]
            outputs.append(self.model(chunk, encode=True).cpu())
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def _predict_init_batch(self, batch):
        global_orients, body_poses, betas = [], [], []
        for start in range(0, len(batch), self.max_batch_size):
            chunk = batch[start:start + self.max_batch_size]
            pred_global_orient, pred_body_pose, pred_betas, _ = self.model(chunk, encode=False)
            global_orients.append(pred_global_orient.cpu())
            body_poses.append(pred_body_pose.cpu())
            betas.append(pred_betas.cpu())
        return (
            torch.cat(global_orients, dim=0),
            torch.cat(body_poses, dim=0),
            torch.cat(betas, dim=0),
        )
    
    @torch.no_grad()
    def run(self, video, tracking_results, patch_h=256, patch_w=256):
        
        if isinstance(video, (list, tuple)):
            cap = video
            is_video = False
            length = len(video)
            height, width = cv2.imread(video[0]).shape[:2]
        elif osp.isfile(video):
            cap = cv2.VideoCapture(video)
            is_video = True
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:   # Image list
            cap = video
            is_video = False
            length = len(video)
            height, width = cv2.imread(video[0]).shape[:2]
        
        frame_id = 0
        bar = Bar('Feature extraction ...', fill='#', max=length)
        while True:
            if is_video:
                flag, img = cap.read()
                if not flag:
                    break
            else:
                if frame_id >= len(cap):
                    break
                img = cv2.imread(cap[frame_id])
            subjects = self._iter_frame_subjects(tracking_results, frame_id)
            if subjects:
                batch_imgs = []
                batch_meta = []
                for _id, frame_id2, val in subjects:
                    bbox = val['bbox'][frame_id2]
                    cx, cy, scale = bbox
                    norm_img, crop_img = process_image(img[..., ::-1], [cx, cy], scale, patch_h, patch_w)
                    batch_imgs.append(norm_img)
                    batch_meta.append((_id, frame_id2, bbox, val))

                batch_tensor = torch.from_numpy(np.stack(batch_imgs)).to(self.device)
                features = self._encode_batch(batch_tensor)

                for idx, (feature, (_id, frame_id2, bbox, val)) in enumerate(zip(features, batch_meta)):
                    tracking_results[_id]['features'].append(feature.unsqueeze(0))

                init_indices = [idx for idx, (_, frame_id2, _, _) in enumerate(batch_meta) if frame_id2 == 0]
                if init_indices:
                    init_tensor = batch_tensor[init_indices]
                    pred_global_orient, pred_body_pose, pred_betas = self._predict_init_batch(init_tensor)
                    for out_idx, batch_idx in enumerate(init_indices):
                        _id = batch_meta[batch_idx][0]
                        tracking_results[_id]['init_global_orient'] = pred_global_orient[out_idx:out_idx + 1]
                        tracking_results[_id]['init_body_pose'] = pred_body_pose[out_idx:out_idx + 1]
                        tracking_results[_id]['init_betas'] = pred_betas[out_idx:out_idx + 1]

                if self.flip_eval:
                    flipped_tensor = torch.flip(batch_tensor, (3,))
                    flipped_features = self._encode_batch(flipped_tensor)

                    for feature, (_id, frame_id2, bbox, val) in zip(flipped_features, batch_meta):
                        tracking_results[_id]['flipped_features'].append(feature.unsqueeze(0))
                        tracking_results[_id]['flipped_bbox'].append(flip_bbox(bbox, width, height))
                        keypoints = val['keypoints'][frame_id2]
                        tracking_results[_id]['flipped_keypoints'].append(flip_kp(keypoints, width))

                    if init_indices:
                        flipped_init_tensor = flipped_tensor[init_indices]
                        pred_global_orient, pred_body_pose, pred_betas = self._predict_init_batch(flipped_init_tensor)
                        for out_idx, batch_idx in enumerate(init_indices):
                            _id = batch_meta[batch_idx][0]
                            tracking_results[_id]['flipped_init_global_orient'] = pred_global_orient[out_idx:out_idx + 1]
                            tracking_results[_id]['flipped_init_body_pose'] = pred_body_pose[out_idx:out_idx + 1]
                            tracking_results[_id]['flipped_init_betas'] = pred_betas[out_idx:out_idx + 1]
                    
            bar.next()
            frame_id += 1
        
        return self.process(tracking_results)
    
    def predict_init(self, norm_img, tracking_results, _id, flip_eval=False):
        prefix = 'flipped_' if flip_eval else ''
        
        pred_global_orient, pred_body_pose, pred_betas, _ = self.model(norm_img, encode=False)
        tracking_results[_id][prefix + 'init_global_orient'] = pred_global_orient.cpu()
        tracking_results[_id][prefix + 'init_body_pose'] = pred_body_pose.cpu()
        tracking_results[_id][prefix + 'init_betas'] = pred_betas.cpu()
        return tracking_results
    
    def process(self, tracking_results):
        output = defaultdict(dict)
        
        for _id, results in tracking_results.items():
            
            for key, val in results.items():
                if isinstance(val, list):
                    if isinstance(val[0], torch.Tensor):
                        val = torch.cat(val)
                    elif isinstance(val[0], np.ndarray):
                        val = np.array(val)
                output[_id][key] = val
        
        return output
