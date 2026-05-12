import os
import stat
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

smplx = types.ModuleType("smplx")
smplx_lbs = types.ModuleType("smplx.lbs")
smplx_lbs.vertices2joints = lambda *args, **kwargs: None
sys.modules.setdefault("smplx", smplx)
sys.modules.setdefault("smplx.lbs", smplx_lbs)

lib_models = types.ModuleType("lib.models")
lib_models.build_body_model = lambda *args, **kwargs: None
sys.modules.setdefault("lib.models", lib_models)

lib_world_grounded = types.ModuleType("lib.world_grounded")
lib_world_grounded.__path__ = []
lib_world_grounded_tracks = types.ModuleType("lib.world_grounded.tracks")
lib_world_grounded_tracks.select_track = lambda *args, **kwargs: None
lib_world_grounded_tracks.to_numpy = lambda value: value
sys.modules.setdefault("lib.world_grounded", lib_world_grounded)
sys.modules.setdefault("lib.world_grounded.tracks", lib_world_grounded_tracks)

from scripts.retarget_smpl_to_opensim import run_opensim, write_process_log


def test_write_process_log_creates_parent_and_file(tmp_path):
    path = tmp_path / "nested" / "opensim_ik.log"
    write_process_log(path, "Frame 0 RMS = 0.1")
    assert path.read_text(encoding="utf-8") == "Frame 0 RMS = 0.1"


def test_write_process_log_ignores_none():
    write_process_log(None, "unused")


def test_run_opensim_replays_and_captures_process_output(tmp_path, capsys):
    setup_xml = tmp_path / "setup.xml"
    setup_xml.write_text("<OpenSimDocument />", encoding="utf-8")
    log_path = tmp_path / "logs" / "opensim_ik.log"

    if os.name == "nt":
        fake_opensim = tmp_path / "fake_opensim.cmd"
        fake_opensim.write_text(
            "@echo off\n"
            "echo fake stdout\n"
            "echo fake stderr>&2\n"
            "exit /b 0\n",
            encoding="utf-8",
        )
    else:
        fake_opensim = tmp_path / "fake_opensim"
        fake_opensim.write_text(
            "#!/bin/sh\n"
            "echo fake stdout\n"
            "echo fake stderr >&2\n",
            encoding="utf-8",
        )
        fake_opensim.chmod(fake_opensim.stat().st_mode | stat.S_IXUSR)

    run_opensim(str(fake_opensim), setup_xml, log_path)

    captured = capsys.readouterr()
    assert captured.out == "fake stdout\n"
    assert captured.err == "fake stderr\n"
    assert log_path.read_text(encoding="utf-8") == "fake stdout\nfake stderr\n"
