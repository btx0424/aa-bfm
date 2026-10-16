"""Download, resample, and cache the G1 LAFAN1 retarget.

Interpolated frames are posed with MuJoCo forward kinematics on the
``g1_waist_unlocked`` MJCF. Body poses and root velocities are stored in
``data/lafan1/motions.npz`` and reused when that cache is already complete.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SRot
from scipy.spatial.transform import Slerp

from active_adaptation.assets.humanoids.g1 import MJCF_PATH
from active_adaptation.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

# Unitree G1 CSV layout from lvhaidong/LAFAN1_Retargeting_Dataset (30 Hz).
# Root is xyz + quaternion xyzw, then the 29 actuated joints.
_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_CLIP_NAMES = (
    "dance1_subject1",
    "dance1_subject2",
    "dance1_subject3",
    "dance2_subject1",
    "dance2_subject2",
    "dance2_subject3",
    "dance2_subject4",
    "dance2_subject5",
    "fallAndGetUp1_subject1",
    "fallAndGetUp1_subject4",
    "fallAndGetUp1_subject5",
    "fallAndGetUp2_subject2",
    "fallAndGetUp2_subject3",
    "fallAndGetUp3_subject1",
    "fight1_subject2",
    "fight1_subject3",
    "fight1_subject5",
    "fightAndSports1_subject1",
    "fightAndSports1_subject4",
    "jumps1_subject1",
    "jumps1_subject2",
    "jumps1_subject5",
    "run1_subject2",
    "run1_subject5",
    "run2_subject1",
    "run2_subject4",
    "sprint1_subject2",
    "sprint1_subject4",
    "walk1_subject1",
    "walk1_subject2",
    "walk1_subject5",
    "walk2_subject1",
    "walk2_subject3",
    "walk2_subject4",
    "walk3_subject1",
    "walk3_subject2",
    "walk3_subject3",
    "walk3_subject4",
    "walk3_subject5",
    "walk4_subject1",
)
_SOURCE_FPS = 30
TARGET_FPS = 50
_HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
_HF_MIRROR = "https://hf-mirror.com"
_FK_FIELDS = ("body_pos", "body_quat")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hub_endpoints() -> list[str]:
    configured = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    endpoints = [configured] if configured else ["https://huggingface.co"]
    if _HF_MIRROR not in endpoints:
        endpoints.append(_HF_MIRROR)
    return endpoints


def _endpoint_reachable(endpoint: str, timeout: float = 5.0) -> bool:
    host = urlparse(endpoint).hostname
    if not host:
        return False
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def _stream_resolve(endpoint: str, filename: str, dest: Path) -> None:
    url = f"{endpoint}/datasets/{_HF_REPO}/resolve/main/{filename}"
    target = dest / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "aa-bfm"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(target)


def _call_with_timeout(fn, seconds: float):
    def handler(signum, frame):
        del signum, frame
        raise TimeoutError(f"timed out after {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _download_one(filename: str, dest: Path, endpoints: list[str], skip_hub: set[str]) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    errors: list[str] = []
    for endpoint in endpoints:
        if hf_hub_download is not None and endpoint not in skip_hub:
            print(f"Downloading {filename} via huggingface_hub from {endpoint}", flush=True)
            try:
                _call_with_timeout(
                    lambda: hf_hub_download(
                        repo_id=_HF_REPO,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=dest,
                        endpoint=endpoint,
                        etag_timeout=20,
                    ),
                    45,
                )
                if (dest / filename).is_file():
                    return
                errors.append(f"huggingface_hub {endpoint}: download returned without the file")
            except Exception as exc:
                errors.append(f"huggingface_hub {endpoint}: {exc}")
                print(f"huggingface_hub failed for {filename}: {exc}", flush=True)
                if isinstance(exc, TimeoutError):
                    skip_hub.add(endpoint)
        print(f"Downloading {filename} from {endpoint}/datasets/{_HF_REPO}/resolve/main", flush=True)
        try:
            _stream_resolve(endpoint, filename, dest)
            return
        except Exception as exc:
            errors.append(f"resolve {endpoint}: {exc}")
            print(f"Direct download failed for {filename}: {exc}", flush=True)
    raise RuntimeError(f"Could not download {filename}:\n" + "\n".join(errors))


def _download_csvs(dest: Path) -> None:
    """Fetch each G1 CSV from the Hub, then from a reachable mirror resolve URL."""
    endpoints = []
    for endpoint in _hub_endpoints():
        if _endpoint_reachable(endpoint):
            endpoints.append(endpoint)
        else:
            print(f"Skipping {endpoint}: network unreachable", flush=True)
    if not endpoints:
        raise RuntimeError("No reachable Hugging Face endpoint. Set HF_ENDPOINT to a mirror.")

    missing = [name for name in _CLIP_NAMES if not (dest / "g1" / f"{name}.csv").is_file()]
    skip_hub: set[str] = set()
    for name in missing:
        _download_one(f"g1/{name}.csv", dest, endpoints, skip_hub)


def _slerp(x: np.ndarray, xp: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    slerp = Slerp(xp, SRot.from_quat(quat_wxyz, scalar_first=True))
    return slerp(x).as_quat(scalar_first=True)


def _resample(values: np.ndarray, source_fps: int, target_fps: int, quat: bool = False) -> np.ndarray:
    length = values.shape[0]
    duration = (length - 1) / source_fps
    xp = np.linspace(0.0, duration, length)
    x = np.arange(0.0, duration, 1.0 / target_fps)
    if quat:
        return _slerp(x, xp, values)
    flat = values.reshape(length, -1)
    out = np.stack([np.interp(x, xp, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape(len(x), *values.shape[1:])


def _angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quat = torch.as_tensor(quat_wxyz, dtype=torch.float32)
    dq = quat_mul(quat[1:], quat_conjugate(quat[:-1]))
    angvel = axis_angle_from_quat(dq) / dt
    angvel = torch.cat([angvel, angvel[-1:]], dim=0)
    return angvel.numpy()


def _linear_velocity(position: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.diff(position, axis=0) / dt
    return np.concatenate([velocity, velocity[-1:]], axis=0)


class Dataloader:
    """Download the G1 retarget of LAFAN1 into ``aa-bfm/data/lafan1``."""

    source_fps = _SOURCE_FPS
    target_fps = TARGET_FPS

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else _repo_root() / "data" / "lafan1"
        self.mjcf_path = Path(MJCF_PATH)
        self._model = None
        self._data = None
        self._free_qposadr = 0
        self._joint_qposadr: np.ndarray | None = None
        self.body_names: list[str] = []
        self._body_ids: np.ndarray | None = None
        self._pelvis_index = 0

    @property
    def archive_path(self) -> Path:
        return self.root / "motions.npz"

    def ensure(self) -> Path:
        if self._archive_has_fk():
            return self.archive_path
        self.root.mkdir(parents=True, exist_ok=True)
        if self.archive_path.is_file():
            self._add_fk()
            return self.archive_path
        raw = self.root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        missing = [name for name in _CLIP_NAMES if not (raw / "g1" / f"{name}.csv").is_file()]
        if missing:
            _download_csvs(raw)
        clips = []
        for name in _CLIP_NAMES:
            csv_path = raw / "g1" / f"{name}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing LAFAN1 clip after download: {csv_path}")
            clips.append(self._fk_clip(name, self._load_csv(csv_path), list(_G1_JOINT_NAMES)))
        self._write(clips, list(_G1_JOINT_NAMES), list(_CLIP_NAMES))
        return self.archive_path

    def _archive_has_fk(self) -> bool:
        if not self.archive_path.is_file():
            return False
        with np.load(self.archive_path, allow_pickle=False) as motions:
            return all(field in motions.files for field in _FK_FIELDS)

    def _add_fk(self) -> None:
        motions = np.load(self.archive_path, allow_pickle=False)
        joint_names = [str(name) for name in motions["joint_names"].tolist()]
        clip_names = [str(name) for name in motions["clip_names"].tolist()]
        starts = motions["starts"]
        ends = motions["ends"]
        time_fields = ("root_pos", "root_quat", "joint_pos", "joint_vel")
        clips = []
        for name, start, end in zip(clip_names, starts, ends):
            frame = slice(int(start), int(end))
            clip = {field: motions[field][frame] for field in time_fields}
            clips.append(self._fk_clip(name, clip, joint_names))
        self._write(clips, joint_names, clip_names)

    def _load_csv(self, path: Path) -> dict[str, np.ndarray]:
        rows = np.loadtxt(path, delimiter=",")
        if rows.ndim != 2 or rows.shape[1] != 7 + len(_G1_JOINT_NAMES):
            raise ValueError(
                f"{path} has shape {rows.shape}, expected (*, {7 + len(_G1_JOINT_NAMES)})"
            )
        root_pos = rows[:, 0:3]
        quat_xyzw = rows[:, 3:7]
        quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=-1)
        joint_pos = rows[:, 7:]
        root_pos = _resample(root_pos, self.source_fps, self.target_fps)
        quat_wxyz = _resample(quat_wxyz, self.source_fps, self.target_fps, quat=True)
        joint_pos = _resample(joint_pos, self.source_fps, self.target_fps)
        root_pos = root_pos.copy()
        root_pos[:, 0:2] -= root_pos[0, 0:2]
        dt = 1.0 / self.target_fps
        joint_vel = _linear_velocity(joint_pos, dt)
        return {
            "root_pos": root_pos.astype(np.float32),
            "root_quat": quat_wxyz.astype(np.float32),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
        }

    def _ensure_model(self, joint_names: list[str]) -> None:
        if self._model is not None:
            return
        import mujoco

        spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
        # Foot pairs reference a scene floor geom that is not part of the robot MJCF.
        while len(spec.pairs):
            spec.delete(spec.pairs[0])
        model = spec.compile()
        name_to_address = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            name_to_address[name] = int(model.jnt_qposadr[joint_id])
        missing = [name for name in joint_names if name not in name_to_address]
        if missing:
            raise KeyError(f"Clip joints missing from {self.mjcf_path.name}: {missing}")
        body_names = []
        body_ids = []
        for body_id in range(1, model.nbody):
            body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            body_ids.append(body_id)
        if "pelvis" not in body_names:
            raise KeyError(f"{self.mjcf_path.name} has no pelvis body")
        if "floating_base_joint" not in name_to_address:
            raise KeyError(f"{self.mjcf_path.name} has no floating_base_joint")
        self._model = model
        self._data = mujoco.MjData(model)
        self._free_qposadr = name_to_address["floating_base_joint"]
        self._joint_qposadr = np.asarray([name_to_address[name] for name in joint_names], dtype=np.int32)
        self.body_names = body_names
        self._body_ids = np.asarray(body_ids, dtype=np.int32)
        self._pelvis_index = body_names.index("pelvis")

    def _fk_clip(
        self,
        name: str,
        clip: dict[str, np.ndarray],
        joint_names: list[str],
    ) -> dict[str, np.ndarray]:
        if clip["joint_pos"].shape[1] != len(joint_names):
            raise ValueError(
                f"{name} has {clip['joint_pos'].shape[1]} joints, expected {len(joint_names)}"
            )
        self._ensure_model(joint_names)
        print(f"Forward kinematics: {name} ({clip['root_pos'].shape[0]} frames)", flush=True)
        body_pos, body_quat = self._forward(clip["root_pos"], clip["root_quat"], clip["joint_pos"])
        root_pos = body_pos[:, self._pelvis_index]
        root_quat = body_quat[:, self._pelvis_index]
        dt = 1.0 / self.target_fps
        return {
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
            "root_lin_vel": _linear_velocity(root_pos, dt).astype(np.float32),
            "root_ang_vel": _angular_velocity(root_quat, dt).astype(np.float32),
            "joint_pos": clip["joint_pos"].astype(np.float32),
            "joint_vel": clip["joint_vel"].astype(np.float32),
            "body_pos": body_pos.astype(np.float32),
            "body_quat": body_quat.astype(np.float32),
        }

    def _forward(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        model = self._model
        data = self._data
        addresses = self._joint_qposadr
        body_ids = self._body_ids
        count = root_pos.shape[0]
        body_pos = np.empty((count, len(self.body_names), 3), dtype=np.float64)
        body_quat = np.empty((count, len(self.body_names), 4), dtype=np.float64)
        qpos = data.qpos
        free = self._free_qposadr
        for frame in range(count):
            qpos[free : free + 3] = root_pos[frame]
            qpos[free + 3 : free + 7] = root_quat[frame]
            qpos[addresses] = joint_pos[frame]
            mujoco.mj_kinematics(model, data)
            body_pos[frame] = data.xpos[body_ids]
            body_quat[frame] = data.xquat[body_ids]
        return body_pos, body_quat

    def _write(
        self,
        clips: list[dict[str, np.ndarray]],
        joint_names: list[str],
        clip_names: list[str],
    ) -> None:
        starts = []
        ends = []
        packed = {key: [] for key in clips[0]}
        cursor = 0
        for clip in clips:
            length = clip["root_pos"].shape[0]
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
            for key, value in clip.items():
                packed[key].append(value)
        np.savez(
            self.archive_path,
            joint_names=np.array(joint_names),
            clip_names=np.array(clip_names),
            body_names=np.array(self.body_names),
            fps=np.int64(self.target_fps),
            starts=np.asarray(starts, dtype=np.int64),
            ends=np.asarray(ends, dtype=np.int64),
            **{key: np.concatenate(value, axis=0) for key, value in packed.items()},
        )
        meta = {
            "fps": self.target_fps,
            "joint_names": joint_names,
            "clip_names": clip_names,
            "body_names": self.body_names,
            "mjcf": str(self.mjcf_path),
            "source": f"https://huggingface.co/datasets/{_HF_REPO}",
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
