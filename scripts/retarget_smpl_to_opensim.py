"""Retarget WHAM SMPL output to a MimicMSK OpenSim IK trajectory.

The script mirrors the MuscleMimic retargeting structure at the interface
level: SMPL motion -> 17 retarget target positions -> OpenSim IK assets.
It writes a TRC marker trajectory, a root coordinate MOT, an IK setup XML,
and can optionally run OpenSim to produce the output coordinate MOT.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from smplx.lbs import vertices2joints

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.models import build_body_model
from lib.world_grounded.tracks import select_track


SMPL24_INDEX = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine": 9,
    "L_Knee": 4,
    "R_Knee": 5,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "L_Toe": 10,
    "R_Toe": 11,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
}


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def axis_rotation(mapping_file: Path, mode: str) -> np.ndarray:
    if mode == "identity":
        return np.eye(3, dtype=np.float64)

    mapping = load_yaml(mapping_file)
    r_os_to_mj = np.asarray(
        mapping["frame_alignment"]["opensim_to_mujoco_rotation"], dtype=np.float64
    )
    if mode == "opensim_to_mujoco":
        return r_os_to_mj
    if mode == "mujoco_to_opensim":
        return r_os_to_mj.T
    raise ValueError(f"Unsupported axis conversion: {mode}")


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def expand_betas(betas: np.ndarray, n_frames: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        return np.repeat(betas[None], n_frames, axis=0)
    return betas[:n_frames]


def smpl24_joints_from_wham(record: dict, device: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    pose_key = "pose_world" if "pose_world" in record else "pose"
    trans_key = "trans_world" if "trans_world" in record else "trans"
    if pose_key not in record:
        raise ValueError("WHAM record must contain 'pose' or 'pose_world'.")
    if trans_key not in record:
        raise ValueError("WHAM record must contain 'trans' or 'trans_world'.")

    pose = to_numpy(record[pose_key]).astype(np.float32)
    trans = to_numpy(record[trans_key]).astype(np.float32)[: len(pose)]
    betas = expand_betas(to_numpy(record["betas"]), len(pose))

    all_joints: list[np.ndarray] = []
    for start in range(0, len(pose), chunk_size):
        end = min(start + chunk_size, len(pose))
        smpl = build_body_model(device, batch_size=end - start)
        kwargs = {
            "global_orient": torch.from_numpy(pose[start:end, :3]).float().to(device),
            "body_pose": torch.from_numpy(pose[start:end, 3:]).float().to(device),
            "betas": torch.from_numpy(betas[start:end]).float().to(device),
            "transl": torch.from_numpy(trans[start:end]).float().to(device),
        }
        with torch.no_grad():
            output = smpl.get_output(**kwargs)
            joints = vertices2joints(smpl.J_regressor, output.vertices)
        all_joints.append(joints.detach().cpu().numpy())

    return np.concatenate(all_joints, axis=0), pose[:, :3]


def marker_name_for_point(point: dict) -> str:
    target_type = point["target_type"]
    if target_type == "opensim_marker":
        return point["opensim_marker"]
    return f"{point['key']}_retarget"


def retarget_positions(smpl24: np.ndarray, points: list[dict], r_axis: np.ndarray) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    positions = np.zeros((smpl24.shape[0], len(points), 3), dtype=np.float64)
    for idx, point in enumerate(points):
        smpl_joint = point["smpl_joint"]
        if smpl_joint not in SMPL24_INDEX:
            raise KeyError(f"Unsupported SMPL joint {smpl_joint!r}")
        names.append(marker_name_for_point(point))
        positions[:, idx] = smpl24[:, SMPL24_INDEX[smpl_joint], :]
    positions = np.einsum("ij,tmj->tmi", r_axis, positions)
    return names, positions


def _normalize(vec: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero anatomical axis {name}.")
    return vec / norm


def anatomical_world_to_opensim(smpl24: np.ndarray, num_frames: int = 30) -> np.ndarray:
    """Estimate one sequence-level world-to-OpenSim anatomical rotation.

    OpenSim convention here is +Y up and +Z subject-right. The returned matrix
    maps world vectors into that OpenSim frame.
    """

    n = min(max(1, num_frames), smpl24.shape[0])
    sample = smpl24[:n]
    pelvis = sample[:, SMPL24_INDEX["Pelvis"]]

    up_vec = np.mean(sample[:, SMPL24_INDEX["Head"]] - pelvis, axis=0)
    right_vec = np.mean(
        0.5
        * (
            sample[:, SMPL24_INDEX["R_Hip"]]
            - sample[:, SMPL24_INDEX["L_Hip"]]
            + sample[:, SMPL24_INDEX["R_Shoulder"]]
            - sample[:, SMPL24_INDEX["L_Shoulder"]]
        ),
        axis=0,
    )

    y_axis = _normalize(up_vec, "up")
    right_vec = right_vec - np.dot(right_vec, y_axis) * y_axis
    z_axis = _normalize(right_vec, "right")
    x_axis = _normalize(np.cross(y_axis, z_axis), "forward")
    z_axis = _normalize(np.cross(x_axis, y_axis), "right_reorthogonalized")
    return np.stack([x_axis, y_axis, z_axis], axis=0)


def parse_model_markers(model_path: Path) -> tuple[ET.ElementTree, ET.Element, dict[str, dict]]:
    tree = ET.parse(model_path)
    root = tree.getroot()
    markers: dict[str, dict] = {}
    for marker in root.iter():
        if marker.tag.endswith("Marker"):
            name = marker.attrib.get("name")
            if not name:
                continue
            parent = marker.findtext("socket_parent_frame")
            loc_text = marker.findtext("location")
            loc = np.fromstring(loc_text or "0 0 0", sep=" ", dtype=np.float64)
            markers[name] = {"element": marker, "parent": parent, "location": loc}
    return tree, root, markers


def add_virtual_markers(config: dict, model_path: Path, out_model: Path) -> Path:
    points = config["retarget_points"]
    needs_virtual = any(p["target_type"] != "opensim_marker" for p in points)
    if not needs_virtual:
        shutil.copy2(model_path, out_model)
        return out_model

    tree, root, markers = parse_model_markers(model_path)
    marker_set = None
    for elem in root.iter():
        if elem.tag.endswith("MarkerSet"):
            marker_set = elem
            break
    if marker_set is None:
        raise ValueError(f"No MarkerSet found in {model_path}")
    objects = marker_set.find("objects")
    if objects is None:
        raise ValueError(f"No MarkerSet/objects found in {model_path}")

    existing = set(markers)
    for point in points:
        if point["target_type"] == "opensim_marker":
            continue
        name = marker_name_for_point(point)
        if name in existing:
            continue

        if point["target_type"] == "opensim_body_origin":
            parent = f"/bodyset/{point['opensim_body']}"
            location = np.zeros(3, dtype=np.float64)
        elif point["target_type"] == "average_opensim_markers":
            source_markers = [markers[m] for m in point["opensim_markers"]]
            parents = {m["parent"] for m in source_markers}
            if len(parents) != 1:
                raise ValueError(f"{point['key']} average markers must share one parent body.")
            parent = source_markers[0]["parent"]
            location = np.mean([m["location"] for m in source_markers], axis=0)
        else:
            raise ValueError(f"Unsupported target_type {point['target_type']!r}")

        marker = ET.SubElement(objects, "Marker", {"name": name})
        ET.SubElement(marker, "components")
        ET.SubElement(marker, "socket_parent_frame").text = parent
        ET.SubElement(marker, "location").text = " ".join(f"{v:.8g}" for v in location)
        ET.SubElement(marker, "fixed").text = "true"

    out_model.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_model, encoding="utf-8", xml_declaration=True)
    return out_model


def ensure_geometry_assets(model_file: Path, out_dir: Path) -> None:
    source_geometry = model_file.parent / "Geometry"
    if not source_geometry.exists():
        return

    target_geometry = out_dir / "Geometry"
    if target_geometry.exists():
        return

    shutil.copytree(source_geometry, target_geometry)


def write_trc(path: Path, marker_names: list[str], positions: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames, n_markers, _ = positions.shape
    times = np.arange(n_frames, dtype=np.float64) / fps

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.8g}\t{fps:.8g}\t{n_frames}\t{n_markers}\tm\t{fps:.8g}\t1\t{n_frames}\n")
        f.write("Frame#\tTime")
        for name in marker_names:
            f.write(f"\t{name}\t\t")
        f.write("\n")
        f.write("\t")
        for i in range(n_markers):
            f.write(f"\tX{i + 1}\tY{i + 1}\tZ{i + 1}")
        f.write("\n")
        for frame, time in enumerate(times, start=1):
            f.write(f"{frame}\t{time:.8f}")
            for point in positions[frame - 1]:
                f.write(f"\t{point[0]:.8f}\t{point[1]:.8f}\t{point[2]:.8f}")
            f.write("\n")


def root_coordinates_from_smpl(
    pelvis_positions: np.ndarray,
    root_rotvec: np.ndarray,
    r_axis: np.ndarray,
) -> np.ndarray:
    root_xyz = np.einsum("ij,tj->ti", r_axis, pelvis_positions)
    root_mats = Rotation.from_rotvec(root_rotvec).as_matrix()
    root_mats_os = np.einsum("ij,tjk,lk->til", r_axis, root_mats, r_axis)
    root_euler = Rotation.from_matrix(root_mats_os).as_euler("xyz", degrees=False)
    return np.concatenate([root_xyz, root_euler], axis=1)


def write_mot(path: Path, columns: list[str], data: np.ndarray, fps: float, in_degrees: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = data.shape[0]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{path.stem}\n")
        f.write("version=1\n")
        f.write(f"nRows={n_frames}\n")
        f.write(f"nColumns={len(columns) + 1}\n")
        f.write(f"inDegrees={'yes' if in_degrees else 'no'}\n")
        f.write("\n")
        f.write("# SIMM Motion File Header:\n")
        f.write(f"name {path.stem}\n")
        f.write(f"datacolumns {len(columns) + 1}\n")
        f.write(f"datarows {n_frames}\n")
        f.write("otherdata 1\n")
        f.write("range 0 {:.8f}\n".format((n_frames - 1) / fps if n_frames else 0.0))
        f.write("endheader\n")
        f.write("time\t" + "\t".join(columns) + "\n")
        for i in range(n_frames):
            row = [i / fps, *data[i]]
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")


def read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_idx = lines.index("endheader")
    except ValueError as exc:
        raise ValueError(f"Could not find endheader in MOT file: {path}") from exc
    columns = lines[header_idx + 1].split()
    rows = [[float(value) for value in line.split()] for line in lines[header_idx + 2 :] if line.strip()]
    return columns, np.asarray(rows, dtype=np.float64)


def root_column_delta(reference_root_mot: Path, ik_mot: Path, num_frames: int) -> dict[str, float]:
    root_cols, root_data = read_mot(reference_root_mot)
    ik_cols, ik_data = read_mot(ik_mot)
    n = min(num_frames, len(root_data), len(ik_data))
    if n <= 0:
        raise ValueError("Cannot calibrate root offset from empty MOT data.")

    deltas: dict[str, float] = {}
    for coord in ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]:
        if coord not in root_cols or coord not in ik_cols:
            raise ValueError(f"Missing {coord} in root calibration MOT files.")
        root_values = root_data[:n, root_cols.index(coord)]
        ik_values = ik_data[:n, ik_cols.index(coord)]
        deltas[coord] = float(np.mean(ik_values - root_values))
    return deltas


def apply_root_delta(root_mot: Path, deltas: dict[str, float]) -> None:
    columns, data = read_mot(root_mot)
    adjusted = data[:, 1:].copy()
    data_columns = columns[1:]
    for coord, delta in deltas.items():
        adjusted[:, data_columns.index(coord)] += delta
    write_mot(root_mot, data_columns, adjusted, fps=1.0 / np.mean(np.diff(data[:, 0])) if len(data) > 1 else 30.0)


def write_ik_setup(
    path: Path,
    model_file: Path,
    marker_file: Path,
    coordinate_file: Path,
    output_motion_file: Path,
    marker_names: list[str],
    points: list[dict],
    fps: float,
    n_frames: int,
    fix_root: bool,
) -> None:
    root = ET.Element("OpenSimDocument", {"Version": "40000"})
    tool = ET.SubElement(root, "InverseKinematicsTool", {"name": "smpl_to_mimicmsk_opensim_ik"})
    ET.SubElement(tool, "results_directory").text = str(output_motion_file.parent)
    ET.SubElement(tool, "model_file").text = str(model_file)
    ET.SubElement(tool, "constraint_weight").text = "Inf"
    ET.SubElement(tool, "accuracy").text = "1e-5"
    task_set = ET.SubElement(tool, "IKTaskSet", {"name": "ik_tasks"})
    objects = ET.SubElement(task_set, "objects")

    for name, point in zip(marker_names, points):
        task = ET.SubElement(objects, "IKMarkerTask", {"name": name})
        ET.SubElement(task, "apply").text = "true"
        ET.SubElement(task, "weight").text = str(point.get("position_weight", 1.0))

    if fix_root:
        for coord in ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]:
            task = ET.SubElement(objects, "IKCoordinateTask", {"name": coord})
            ET.SubElement(task, "apply").text = "true"
            ET.SubElement(task, "value_type").text = "from_file"
            ET.SubElement(task, "value").text = "0"
            ET.SubElement(task, "weight").text = "10000"

    ET.SubElement(task_set, "groups")
    ET.SubElement(tool, "marker_file").text = str(marker_file)
    ET.SubElement(tool, "coordinate_file").text = str(coordinate_file if fix_root else "")
    ET.SubElement(tool, "time_range").text = f"0 {((n_frames - 1) / fps if n_frames else 0.0):.8f}"
    ET.SubElement(tool, "output_motion_file").text = str(output_motion_file)
    ET.SubElement(tool, "report_errors").text = "true"

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def find_opensim_cmd(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for candidate in [
        shutil.which("opensim-cmd"),
        shutil.which("opensim-cmd.exe"),
        r"C:\Users\yangfeiyang\Downloads\OpenSim 4.5\bin\opensim-cmd.exe",
    ]:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def run_opensim(opensim_cmd: str, setup_xml: Path) -> None:
    if not is_ascii_path(setup_xml):
        raise ValueError(
            "OpenSim 4.5 on Windows cannot reliably read non-ASCII setup/model paths. "
            f"Use an ASCII --out-dir, current setup path is: {setup_xml}"
        )
    cmd = [opensim_cmd, "run-tool", str(setup_xml)]
    subprocess.run(cmd, cwd=str(setup_xml.parent), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smpl_pkl", help="WHAM wham_output.pkl or canonical_wham_output.pkl.")
    parser.add_argument("--config", default="configs/retarget/smpl_to_mimicmsk_opensim.yaml")
    parser.add_argument(
        "--track-id",
        default="merge",
        help="Track id to retarget. Use 'merge' to merge ID switches, or 'longest' to use the longest track.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug limit on frames.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--axis-conversion",
        choices=["mujoco_to_opensim", "opensim_to_mujoco", "identity"],
        default="mujoco_to_opensim",
        help="Coordinate-frame conversion applied to SMPL target positions.",
    )
    parser.add_argument(
        "--alignment",
        choices=["mapping", "anatomical"],
        default="anatomical",
        help="Use model_mapping axes or estimate a sequence anatomical OpenSim frame from SMPL joints.",
    )
    parser.add_argument("--run-ik", action="store_true", help="Run OpenSim IK after writing assets.")
    parser.add_argument("--free-root", action="store_true", help="Diagnostic mode: do not lock root coordinates in IK.")
    parser.add_argument(
        "--root-calibration",
        choices=["none", "constant_from_free_ik"],
        default="constant_from_free_ik",
        help="Calibrate fixed OpenSim root coordinates before the final fixed-root IK pass.",
    )
    parser.add_argument("--root-calib-frames", type=int, default=10)
    parser.add_argument("--opensim-cmd", default=None)
    args = parser.parse_args()

    smpl_pkl = Path(args.smpl_pkl).resolve()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    references = config["references"]
    model_file = resolve_path(config_path.parent, references["opensim_model"])
    mapping_file = resolve_path(config_path.parent, references["opensim_mujoco_mapping"])

    results = joblib.load(smpl_pkl)
    track_id, record = select_track(results, args.track_id)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else smpl_pkl.parent / "opensim_retarget" / f"track_{track_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    smpl24, root_rotvec = smpl24_joints_from_wham(record, args.device, args.chunk_size)
    if args.max_frames is not None:
        smpl24 = smpl24[: args.max_frames]
        root_rotvec = root_rotvec[: args.max_frames]
    if args.alignment == "anatomical":
        r_axis = anatomical_world_to_opensim(smpl24)
    else:
        r_axis = axis_rotation(mapping_file, args.axis_conversion)
    points = config["retarget_points"]
    marker_names, marker_positions = retarget_positions(smpl24, points, r_axis)

    ensure_geometry_assets(model_file, out_dir)
    ik_model = add_virtual_markers(config, model_file, out_dir / "MimicMSK_OpenSim_retarget_markers.osim")
    trc_file = out_dir / "smpl_17pt_markers.trc"
    root_mot = out_dir / "wham_fixed_root.mot"
    output_mot = out_dir / "opensim_ik.mot"
    setup_xml = out_dir / "smpl_to_opensim_ik_setup.xml"

    write_trc(trc_file, marker_names, marker_positions, args.fps)
    root_data = root_coordinates_from_smpl(smpl24[:, SMPL24_INDEX["Pelvis"], :], root_rotvec, r_axis)
    root_mot_data = root_data.copy()
    root_mot_data[:, 3:] = np.rad2deg(root_mot_data[:, 3:])
    write_mot(root_mot, ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"], root_mot_data, args.fps)
    final_fix_root = config.get("root_policy", {}).get("root_fixed", True) and not args.free_root
    write_ik_setup(
        setup_xml,
        ik_model,
        trc_file,
        root_mot,
        output_mot,
        marker_names,
        points,
        args.fps,
        marker_positions.shape[0],
        fix_root=final_fix_root,
    )

    print(f"track_id: {track_id}")
    print(f"frames: {marker_positions.shape[0]}")
    print(f"retarget_points: {len(marker_names)}")
    print(f"ik_model: {ik_model}")
    print(f"marker_trc: {trc_file}")
    print(f"root_mot: {root_mot}")
    print(f"ik_setup: {setup_xml}")

    if args.run_ik:
        opensim_cmd = find_opensim_cmd(args.opensim_cmd)
        if not opensim_cmd:
            raise FileNotFoundError("Could not find opensim-cmd. Pass --opensim-cmd explicitly.")
        if final_fix_root and args.root_calibration == "constant_from_free_ik":
            calibration_motion = out_dir / "root_calibration_free_ik.mot"
            calibration_setup = out_dir / "root_calibration_free_ik_setup.xml"
            calibration_trc = out_dir / "root_calibration_markers.trc"
            calibration_root_mot = out_dir / "root_calibration_wham_root.mot"
            calibration_frames = min(max(1, args.root_calib_frames), marker_positions.shape[0])
            write_trc(calibration_trc, marker_names, marker_positions[:calibration_frames], args.fps)
            write_mot(
                calibration_root_mot,
                ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"],
                root_mot_data[:calibration_frames],
                args.fps,
            )
            write_ik_setup(
                calibration_setup,
                ik_model,
                calibration_trc,
                calibration_root_mot,
                calibration_motion,
                marker_names,
                points,
                args.fps,
                calibration_frames,
                fix_root=False,
            )
            run_opensim(opensim_cmd, calibration_setup)
            deltas = root_column_delta(calibration_root_mot, calibration_motion, calibration_frames)
            apply_root_delta(root_mot, deltas)
            print("root_calibration_delta:")
            for key in ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]:
                print(f"  {key}: {deltas[key]:.8f}")
        run_opensim(opensim_cmd, setup_xml)
        print(f"output_mot: {output_mot}")
    else:
        print(f"output_mot: {output_mot} (not generated; rerun with --run-ik)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
