from __future__ import annotations

import json
import os
import os.path as osp
import urllib.request
import zipfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module

import cv2
import numpy as np
import scipy.signal as signal
import torch

from ultralytics import YOLO

warnings.filterwarnings(
    "ignore",
    message=r"Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*",
    category=UserWarning,
)

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30
MINIMUM_JOINTS = 6

DEFAULT_RTMPOSE_MODELS = {
    "rtmpose-m": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    },
    "rtmpose-t": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
    },
}


def _load_mmpose_api(name):
    try:
        return getattr(import_module("mmpose.apis"), name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mmpose is required only for --pose-backend vitpose. "
            "Install mmpose/mmdet/mmcv for ViTPose, or run with --pose-backend rtmpose."
        ) from exc


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = np.asarray(a[:4], dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(b[:4], dtype=np.float32)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


def _simple_get_track_id(pose_results, pose_results_last, next_id, tracking_thr=TRACKING_THR):
    used_last = set()
    for pose_result in pose_results:
        best_idx = None
        best_iou = tracking_thr
        for idx, last in enumerate(pose_results_last):
            if idx in used_last or "track_id" not in last:
                continue
            iou = _bbox_iou_xyxy(pose_result["bbox"], last["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is None:
            pose_result["track_id"] = next_id
            next_id += 1
        else:
            pose_result["track_id"] = pose_results_last[best_idx]["track_id"]
            used_last.add(best_idx)

    return pose_results, next_id


@contextmanager
def _torch_load_weights_only_compat():
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


class RTMPoseONNXModel:
    def __init__(self, model_source, device):
        import onnxruntime as ort

        self.model_path = self._resolve_model_source(model_source)
        self.device = device.lower()

        providers = ["CPUExecutionProvider"]
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.input_w = 192
        self.input_h = 256
        self.padding = 1.25
        self.simcc_split_ratio = 2.0
        self.to_rgb = True
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self._load_pipeline_metadata()

        self.input_size = np.array([self.input_w, self.input_h], dtype=np.float32)

    def _resolve_model_source(self, model_source):
        if model_source is None:
            model_source = "rtmpose-m"

        model_source = str(model_source)
        alias = model_source.lower()
        if alias in DEFAULT_RTMPOSE_MODELS:
            return self._ensure_official_model(alias)

        if osp.isdir(model_source):
            found = self._find_end2end_onnx(model_source)
            if found is not None:
                return found

        if osp.isabs(model_source) and osp.exists(model_source):
            return model_source

        repo_relative = osp.join(ROOT_DIR, model_source)
        if osp.isdir(repo_relative):
            found = self._find_end2end_onnx(repo_relative)
            if found is not None:
                return found
        if osp.exists(repo_relative):
            return repo_relative

        checkpoints_relative = osp.join(ROOT_DIR, "checkpoints", model_source)
        if osp.isdir(checkpoints_relative):
            found = self._find_end2end_onnx(checkpoints_relative)
            if found is not None:
                return found
        if osp.exists(checkpoints_relative):
            return checkpoints_relative

        raise FileNotFoundError(f"RTMPose model not found: {model_source}")

    def _ensure_official_model(self, alias):
        url = DEFAULT_RTMPOSE_MODELS[alias]["url"]
        target_dir = osp.join(ROOT_DIR, "checkpoints", "rtmpose", alias)
        os.makedirs(target_dir, exist_ok=True)

        onnx_path = self._find_end2end_onnx(target_dir)
        if onnx_path is not None:
            return onnx_path

        zip_path = osp.join(target_dir, osp.basename(url))
        if not osp.exists(zip_path):
            urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(target_dir)

        onnx_path = self._find_end2end_onnx(target_dir)
        if onnx_path is None:
            raise FileNotFoundError(f"Failed to locate end2end.onnx after extracting {zip_path}")
        return onnx_path

    @staticmethod
    def _find_end2end_onnx(root_dir):
        for dirpath, _, filenames in os.walk(root_dir):
            if "end2end.onnx" in filenames:
                return osp.join(dirpath, "end2end.onnx")
        return None

    def _load_pipeline_metadata(self):
        pipeline_path = osp.join(osp.dirname(self.model_path), "pipeline.json")
        if not osp.exists(pipeline_path):
            return

        with open(pipeline_path, "r", encoding="utf-8") as file:
            pipeline = json.load(file)

        tasks = pipeline.get("pipeline", {}).get("tasks", [])
        for task in tasks:
            if task.get("name") == "Preprocess":
                for transform in task.get("transforms", []):
                    if transform.get("type") == "TopDownGetBboxCenterScale":
                        self.padding = float(transform.get("padding", self.padding))
                        image_size = transform.get("image_size", [self.input_w, self.input_h])
                        self.input_w = int(image_size[0])
                        self.input_h = int(image_size[1])
                    elif transform.get("type") == "Normalize":
                        self.mean = np.asarray(transform.get("mean", self.mean), dtype=np.float32)
                        self.std = np.asarray(transform.get("std", self.std), dtype=np.float32)
                        self.to_rgb = bool(transform.get("to_rgb", self.to_rgb))

            if task.get("name") == "postprocess":
                params = task.get("params", {})
                self.simcc_split_ratio = float(params.get("simcc_split_ratio", self.simcc_split_ratio))

    def infer(self, img, person_results):
        if not person_results:
            return []

        bboxes = np.asarray([person_result["bbox"][:4] for person_result in person_results], dtype=np.float32)
        inputs, centers, scales = self._preprocess(img, bboxes)
        simcc_x, simcc_y = self.session.run(self.output_names, {self.input_name: inputs})
        keypoints, scores = self._postprocess(simcc_x, simcc_y, centers, scales)

        pose_results = []
        for bbox, pose_keypoints, pose_scores in zip(bboxes, keypoints, scores):
            pose_results.append(
                {
                    "bbox": np.concatenate((bbox.astype(np.float32), np.array([1.0], dtype=np.float32))),
                    "keypoints": np.concatenate((pose_keypoints, pose_scores[:, None]), axis=-1).astype(np.float32),
                }
            )
        return pose_results

    def _preprocess(self, img, bboxes):
        processed = []
        centers = []
        scales = []

        for bbox in bboxes:
            center, scale = self._bbox_xyxy2cs(bbox, padding=self.padding)
            crop, scale = self._top_down_affine(img, scale, center)
            if self.to_rgb:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = (crop.astype(np.float32) - self.mean) / self.std
            processed.append(crop.transpose(2, 0, 1))
            centers.append(center)
            scales.append(scale)

        inputs = np.asarray(processed, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32)
        return inputs, centers, scales

    def _postprocess(self, simcc_x, simcc_y, centers, scales):
        keypoints, scores = self._decode(simcc_x, simcc_y)
        keypoints = keypoints / self.input_size[None, None, :] * scales[:, None, :]
        keypoints = keypoints + centers[:, None, :] - scales[:, None, :] / 2.0
        return keypoints.astype(np.float32), scores.astype(np.float32)

    def _decode(self, simcc_x, simcc_y):
        batch_size, num_keypoints, _ = simcc_x.shape
        simcc_x_flat = simcc_x.reshape(batch_size * num_keypoints, -1)
        simcc_y_flat = simcc_y.reshape(batch_size * num_keypoints, -1)

        x_locs = np.argmax(simcc_x_flat, axis=1)
        y_locs = np.argmax(simcc_y_flat, axis=1)
        max_val_x = np.max(simcc_x_flat, axis=1)
        max_val_y = np.max(simcc_y_flat, axis=1)
        vals = np.minimum(max_val_x, max_val_y)

        keypoints = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        keypoints[vals <= 0.0] = -1.0
        keypoints /= self.simcc_split_ratio

        keypoints = keypoints.reshape(batch_size, num_keypoints, 2)
        vals = vals.reshape(batch_size, num_keypoints)
        return keypoints, vals

    @staticmethod
    def _bbox_xyxy2cs(bbox, padding=1.25):
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        scale = np.array([(x2 - x1), (y2 - y1)], dtype=np.float32) * float(padding)
        return center, scale

    @staticmethod
    def _fix_aspect_ratio(scale, aspect_ratio):
        width = scale[0]
        height = scale[1]
        if width > height * aspect_ratio:
            height = width / aspect_ratio
        else:
            width = height * aspect_ratio
        return np.array([width, height], dtype=np.float32)

    @staticmethod
    def _rotate_point(point, angle_rad):
        sin_value = np.sin(angle_rad)
        cos_value = np.cos(angle_rad)
        rotation = np.array([[cos_value, -sin_value], [sin_value, cos_value]], dtype=np.float32)
        return rotation @ point

    @staticmethod
    def _get_3rd_point(point_a, point_b):
        direction = point_a - point_b
        return point_b + np.array([-direction[1], direction[0]], dtype=np.float32)

    def _get_warp_matrix(self, center, scale, rot, output_size):
        src_w = scale[0]
        dst_w, dst_h = output_size
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
        dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0] = center
        src[1] = center + src_dir
        src[2] = self._get_3rd_point(src[0], src[1])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst[1] = dst[0] + dst_dir
        dst[2] = self._get_3rd_point(dst[0], dst[1])

        return cv2.getAffineTransform(np.float32(src), np.float32(dst))

    def _top_down_affine(self, img, scale, center):
        scale = self._fix_aspect_ratio(scale, aspect_ratio=self.input_w / self.input_h)
        warp_matrix = self._get_warp_matrix(center, scale, 0.0, (self.input_w, self.input_h))
        crop = cv2.warpAffine(img, warp_matrix, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        return crop, scale


class DetectionModel:
    def __init__(
        self,
        device,
        bbox_model_ckpt=None,
        bbox_imgsz=640,
        detect_scale=1.0,
        pose_backend="vitpose",
        pose_model_ckpt=None,
        min_track_frames=MINIMUM_FRMAES,
    ):
        self.device = device
        self.pose_backend = str(pose_backend or "vitpose").lower()

        if self.pose_backend == "vitpose":
            init_pose_model = _load_mmpose_api("init_pose_model")
            pose_model_cfg = osp.join(
                VIT_DIR, "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py"
            )
            pose_model_ckpt = self._resolve_model_source(
                pose_model_ckpt, default_path=osp.join(ROOT_DIR, "checkpoints", "vitpose-h-multi-coco.pth")
            )
            self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        elif self.pose_backend == "rtmpose":
            self.pose_model = RTMPoseONNXModel(pose_model_ckpt, device=device)
        else:
            raise ValueError(f"Unsupported pose backend: {pose_backend}")

        bbox_model_ckpt = self._resolve_model_source(
            bbox_model_ckpt, default_path=osp.join(ROOT_DIR, "checkpoints", "yolov8x.pt")
        )
        # Ultralytics still calls torch.load() without weights_only=... in some
        # code paths, so patch the default just around checkpoint loading.
        with _torch_load_weights_only_compat():
            self.bbox_model = YOLO(bbox_model_ckpt)

        self.bbox_imgsz = int(bbox_imgsz)
        self.detect_scale = float(detect_scale)
        self.min_track_frames = int(min_track_frames)
        self.initialize_tracking()

    def _resolve_model_source(self, model_source, default_path=None):
        if model_source is None:
            return default_path

        if osp.isabs(model_source) and osp.exists(model_source):
            return model_source

        repo_relative = osp.join(ROOT_DIR, model_source)
        if osp.exists(repo_relative):
            return repo_relative

        checkpoints_relative = osp.join(ROOT_DIR, "checkpoints", model_source)
        if osp.exists(checkpoints_relative):
            return checkpoints_relative

        return model_source

    def _resize_for_detection(self, img):
        if self.detect_scale >= 0.999:
            return img, 1.0

        height, width = img.shape[:2]
        resized = cv2.resize(
            img,
            (max(1, int(round(width * self.detect_scale))), max(1, int(round(height * self.detect_scale)))),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, self.detect_scale

    def _restore_original_scale(self, pose_results, scale):
        if scale >= 0.999:
            return pose_results

        inv_scale = 1.0 / scale
        for pose_result in pose_results:
            pose_result["bbox"] = np.array(pose_result["bbox"], copy=True)
            pose_result["bbox"][:4] *= inv_scale
            pose_result["keypoints"] = np.array(pose_result["keypoints"], copy=True)
            pose_result["keypoints"][:, :2] *= inv_scale
        return pose_results

    def _infer_pose(self, img, person_results):
        if self.pose_backend == "vitpose":
            inference_top_down_pose_model = _load_mmpose_api("inference_top_down_pose_model")
            pose_results, _ = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results=person_results,
                format="xyxy",
                return_heatmap=False,
                outputs=None,
            )
            return pose_results
        return self.pose_model.infer(img, person_results)

    def _assign_track_ids(self, pose_results, fps):
        try:
            get_track_id = _load_mmpose_api("get_track_id")
        except ModuleNotFoundError:
            return _simple_get_track_id(
                pose_results,
                self.pose_results_last,
                self.next_id,
                tracking_thr=TRACKING_THR,
            )

        return get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=TRACKING_THR,
            use_one_euro=True,
            fps=fps,
        )

    def initialize_tracking(self):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {"id": [], "frame_id": [], "bbox": [], "keypoints": []}

    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])

    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results["keypoints"].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, mask_i) in enumerate(zip(X, mask)):
            bb = [kp[mask_i, 0].min(), kp[mask_i, 1].min(), kp[mask_i, 0].max(), kp[mask_i, 1].max()]
            cx, cy = [(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            side = np.stack((bb_w, bb_h)).max()
            bbox[i] = np.array((cx, cy, side))

        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results["bbox"] = bbox

    def track(self, img, fps, length):
        img_for_detection, detect_scale = self._resize_for_detection(img)

        bbox_array = self.bbox_model.predict(
            img_for_detection,
            device=self.device,
            classes=0,
            conf=BBOX_CONF,
            imgsz=self.bbox_imgsz,
            save=False,
            verbose=False,
        )[0].boxes.xyxy.detach().cpu().numpy()
        person_results = [{"bbox": bbox} for bbox in bbox_array]

        pose_results = self._infer_pose(img_for_detection, person_results)
        pose_results = self._restore_original_scale(pose_results, detect_scale)

        pose_results, self.next_id = self._assign_track_ids(pose_results, fps)

        for pose_result in pose_results:
            n_valid = (pose_result["keypoints"][:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS:
                continue

            track_id = pose_result["track_id"]
            xyxy = pose_result["bbox"]
            bbox = self.xyxy_to_cxcys(xyxy)

            self.tracking_results["id"].append(track_id)
            self.tracking_results["frame_id"].append(self.frame_id)
            self.tracking_results["bbox"].append(bbox)
            self.tracking_results["keypoints"].append(pose_result["keypoints"])

        self.frame_id += 1
        self.pose_results_last = pose_results

    def process(self, fps):
        for key in ["id", "frame_id", "keypoints"]:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        self.compute_bboxes_from_keypoints()

        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results["id"])
        for track_id in ids:
            indices = np.where(self.tracking_results["id"] == track_id)[0]
            for key, value in self.tracking_results.items():
                if key == "id":
                    continue
                output[track_id][key] = value[indices]

        ids = list(output.keys())
        for track_id in ids:
            if len(output[track_id]["bbox"]) < self.min_track_frames:
                del output[track_id]
                continue

            kernel = int(int(fps / 2) / 2) * 2 + 1
            max_kernel = len(output[track_id]["bbox"])
            if max_kernel % 2 == 0:
                max_kernel -= 1

            if max_kernel >= 3:
                kernel = min(kernel, max_kernel)
                smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[track_id]["bbox"].T]).T
                output[track_id]["bbox"] = smoothed_bbox

        return output
