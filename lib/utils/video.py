import os.path as osp

import cv2
from loguru import logger


def _normalize_rotation(rotation):
    rotation = int(round(float(rotation))) % 360
    return rotation


def _rotation_to_cv2_code(rotation):
    rotation = _normalize_rotation(rotation)
    if rotation == 0:
        return None
    if rotation == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if rotation == 180:
        return cv2.ROTATE_180
    if rotation == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    raise ValueError(f'Unsupported video rotation metadata: {rotation}')


def get_video_rotation(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video file {video_path}')

    try:
        if not hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):
            return 0
        return _normalize_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    finally:
        cap.release()


def normalize_video_rotation(video_path, output_dir, output_name='_rotation_fixed_input.mp4'):
    """Bake video rotation metadata into pixels using OpenCV only.

    Some phone videos are stored in landscape pixel order and rely on rotation
    metadata for correct display. WHAM mixes multiple video readers, so we
    rewrite those inputs to a metadata-free MP4 before preprocessing.
    """
    rotation = get_video_rotation(video_path)
    if rotation == 0:
        logger.info('No rotation metadata detected, using original video')
        return video_path

    rotate_code = _rotation_to_cv2_code(rotation)
    fixed_path = osp.join(output_dir, output_name)
    logger.info(f'Video has rotation={rotation} deg, baking it into pixels')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video file {video_path}')

    if hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f'Failed to read the first frame from {video_path}')

    frame = cv2.rotate(frame, rotate_code)
    frame_height, frame_width = frame.shape[:2]

    writer = cv2.VideoWriter(
        fixed_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Failed to create normalized video {fixed_path}')

    try:
        writer.write(frame)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.rotate(frame, rotate_code))
    finally:
        writer.release()
        cap.release()

    logger.info(f'Rotation-normalized video saved to {fixed_path}')
    return fixed_path
